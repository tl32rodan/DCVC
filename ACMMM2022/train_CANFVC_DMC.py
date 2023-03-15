import argparse
import os
import csv
from functools import partial

import yaml
import comet_ml
import flowiz as fz
import numpy as np
import json
import torch
import torch.nn.functional as F

from torchinfo import summary
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from pytorch_msssim import MS_SSIM

from src.models.image_model import IntraNoAR

from dataloader import VimeoDataset, VideoTestData
from src.utils.psnr import mse2psnr
from src.utils.stream_helper import get_padding_size, get_state_dict
from src.utils.vision import PlotFlow, PlotHeatMap, save_image, Alignment

from ptflops import get_model_complexity_info


plot_flow = PlotFlow().cuda()
plot_bitalloc = PlotHeatMap("RB").cuda()

phase = {'trainMV': -1, 
         'trainMC': -1,
         'trainRes_2frames_RecOnly': -1,
         'trainRes_2frames': -1, 
         'trainAll_2frames': -1, 
         'trainAll_fullgop': -1, 
         'trainAll_RNN_1': 10, 
         'trainAll_RNN_2': 12,
         'train_aux': 100}

class CompressesModel(LightningModule):
    """Basic Compress Model"""

    def __init__(self):
        super(CompressesModel, self).__init__()

    def named_main_parameters(self, prefix=''):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' not in name:
                yield (name, param)

    def main_parameters(self):
        for _, param in self.named_main_parameters():
            yield param

    def named_aux_parameters(self, prefix=''):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' in name:
                yield (name, param)


class Pframe(CompressesModel):
    def __init__(self, args):
        super(Pframe, self).__init__()
        self.args = args
        self.criterion = nn.MSELoss(reduction='none') if not self.args.ssim else MS_SSIM(data_range=1., size_average=True, channel=3).cuda()

        self.i_model = IntraNoAR()
        self.i_model.eval()
        
        if self.args.inter_coder_type == 'CVAE':
            from src.models.video_model import DMC
        elif self.args.inter_coder_type == 'CANF':
            from src.models.video_model import CANFVC_DMC as DMC
        self.p_model = DMC() 

        self.dpb = None
        self.q_level = {85: 0, 170: 1, 380: 2, 840: 3}[self.args.lmda]

    def load_args(self, args):
        self.args = args

    def forward(self, coding_frame, dpb, order=0, q_level=None):
        if q_level is None:
            q_level = self.q_level

        if order == 0:
            result = self.i_model.forward(coding_frame, self.i_model.q_scale[q_level:q_level+1])
            dpb = {
                "ref_frame": result["x_hat"],
                "ref_feature": None,
                "ref_y": None,
                "ref_mv_y": None,
            }
            recon_frame = result["x_hat"]
            bpp = result["bpp"]
        else:
            result = self.p_model.forward_one_frame(coding_frame, dpb,
                                                mv_y_q_scale=self.p_model.mv_y_q_scale[q_level:q_level+1],
                                                y_q_scale=self.p_model.y_q_scale[q_level:q_level+1])
            dpb = result["dpb"]

            if self.args.no_feat_buffer and dpb['ref_feature'] is not None:
                del dpb['ref_feature'] 
                dpb['ref_feature'] = None

            recon_frame = dpb["ref_frame"]
            bpp = result["bpp"]
         
        return recon_frame, bpp, dpb, result

    def training_step(self, batch, batch_idx):

        def disable_modules(modules):
            for module in modules:
                if not isinstance(module, nn.Parameter):
                    for param in module.parameters(): 
                        self.optimizers().state[param] = {} # remove all state (step, exp_avg, exp_avg_sg)
                else:
                    self.optimizers().state[module] = {} # remove all state (step, exp_avg, exp_avg_sg)

                module.requires_grad_(False)

        def activate_modules(modules):
            for module in modules:
                module.requires_grad_(True)
            
        epoch = self.current_epoch
        if epoch <= phase['trainMV']:
            pass
        elif epoch <= phase['trainMC']:
            pass
        elif epoch <= phase['trainAll_2frames']:
            training_frames = 2
            if epoch <= phase['trainRes_2frames']:
                activate_modules_list = [
                                         #self.p_model.contextual_coder, self.p_model.DQ,
                                         self.p_model.contextual_encoder, self.p_model.contextual_decoder,
                                         self.p_model.contextual_hyper_prior_encoder, self.p_model.contextual_hyper_prior_decoder,
                                         self.p_model.temporal_prior_encoder, self.p_model.y_prior_fusion, self.p_model.y_spatial_prior, 
                                         #self.p_model.mv_y_q_basic, self.p_model.mv_y_q_scale,
                                         #self.p_model.y_q_basic, self.p_model.y_q_scale,
                                        ]
                disable_modules_list = [
                                         #self.p_model.mv_y_q_basic, self.p_model.mv_y_q_scale,
                                         #self.p_model.y_q_basic, self.p_model.y_q_scale,
                                       ]
            else:
                activate_modules_list = [self.p_model]
                disable_modules_list = [ 
                                         #self.p_model.mv_y_q_scale,
                                         #self.p_model.y_q_scale,
                                       ]
        else:
            training_frames = 7
            activate_modules_list = [self.p_model]
            disable_modules_list = [ 
                                     #self.p_model.mv_y_q_scale,
                                     #self.p_model.y_q_scale,
                                   ]
            #disable_modules_list = []
        
        self.requires_grad_(False)
        activate_modules(activate_modules_list)
        disable_modules(disable_modules_list)

        dist_list = []
        rate_list = []
        mc_error_list = []
        
        frame_count = 0
        total_loss = torch.tensor(0., dtype=torch.float, device=batch.device)
        
        for frame_idx in range(0, training_frames):
            frame_count += 1
            if epoch < phase["trainAll_fullgop"] and frame_idx > 0:
                self.dpb['ref_frame'] = self.dpb['ref_frame'].detach()

            #coding_frames = batch[:, frame_idx].chunk(4, dim=0)
            #for q_level, coding_frame in zip([0, 1, 2, 3], coding_frames):

            coding_frame = batch[:, frame_idx]

            rec_frame, bpp, self.dpb, result = self(coding_frame, self.dpb, frame_idx)

            distortion, rate = result["mse"], result["bpp"]
            
            if self.args.ssim:
                distortion = (1 - distortion)/64
            
            loss = self.args.lmda_scale * self.args.lmda * distortion.mean() + rate.mean()

            total_loss += loss
            dist_list.append(distortion.mean())
            rate_list.append(rate.mean())
            #mc_error_list.append(mc_error.mean())
        
        total_loss = total_loss / (training_frames - 1)
        distortion = torch.mean(torch.tensor(dist_list))
        rate = torch.mean(torch.tensor(rate_list))
        #mc_error = torch.mean(torch.tensor(mc_error_list))

        logs = {
                'train/loss': total_loss.item(),
                'train/distortion': distortion.item(), 
                'train/PSNR': mse2psnr(distortion.item()), 
                'train/rate': rate.item(), 
                #'train/mc_error': mc_error.item(),
               }
        #if epoch <= phase['trainRes_2frames_RecOnly']:
        #    logs.update({'train/DQ_distortion': DQ_distortion.mean().item()})

        self.log_dict(logs)
        return total_loss 

    def validation_step(self, batch, batch_idx):
        def create_grid(img):
            return make_grid(torch.unsqueeze(img, 1)).cpu().detach().numpy()[0]

        def upload_img(tnsr, tnsr_name, ch="first", grid=True):
            if grid:
                tnsr = create_grid(tnsr)

            self.logger.experiment.log_image(tnsr, name=tnsr_name, step=self.current_epoch,
                                             image_channels=ch, overwrite=True)

        dataset_name, seq_name, batch, frame_id_start = batch
        frame_id = int(frame_id_start)

        seq_name = seq_name[0]
        dataset_name = dataset_name[0]

        gop_size = batch.size(1)

        height, width = batch.size()[3:]

        psnr_list = []
        mc_psnr_list = []
        mse_list = []
        rate_list = []
        m_rate_list = []
        loss_list = []
        align = Alignment()

        epoch = int(self.current_epoch)

        self.dpb = None
        for frame_idx in range(gop_size):
            if frame_idx != 0:
                coding_frame = batch[:, frame_idx]

                rec_frame, bpp, self.dpb, result = self(align.align(coding_frame), self.dpb, frame_idx)
                rec_frame = align.resume(rec_frame).clamp(0, 1)
                BDQ = align.resume(result["BDQ"]).clamp(0, 1)

                mse, rate, mc_error = self.criterion(rec_frame, coding_frame).mean().item(), result["bpp"].item(), result["x2_mse"].item()
                m_rate = result["bpp_mv_y"].item() + result["bpp_mv_z"].item()


                #if frame_idx <= 2:

                #    flow_hat = align.resume(result["mv_hat"])
                #    flow_rgb = torch.from_numpy(
                #        fz.convert_from_flow(flow_hat[0].permute(1, 2, 0).cpu().numpy()) / 255).permute(2, 0, 1)
                #    upload_img(flow_rgb.cpu().numpy(), f'{seq_name}_{epoch}_dec_flow_{frame_idx}.png', grid=False)
                #    
                #    upload_img(ref_frame.cpu().numpy()[0], f'{seq_name}_{epoch}_ref_frame_{frame_idx}.png', grid=False)
                #    upload_img(coding_frame.cpu().numpy()[0], f'{seq_name}_{epoch}_gt_frame_{frame_idx}.png', grid=False)
                #    upload_img(rec_frame.cpu().numpy()[0], seq_name + '_{:d}_rec_frame_{:d}_{:.3f}.png'.format(epoch, frame_idx, psnr), grid=False)

                if self.args.ssim:
                    psnr = mse
                else:
                    psnr = mse2psnr(mse)
                loss = self.args.lmda_scale * self.args.lmda * mse + rate #+ self.args.lmda * 0.01 * mc_error

                m_rate_list.append(m_rate)
            else:
                coding_frame = batch[:, frame_idx]
                rec_frame, rate, self.dpb, _ = self(align.align(coding_frame), None, frame_idx)

                rec_frame = align.resume(rec_frame).clamp(0, 1)

                mse, rate = self.criterion(rec_frame, coding_frame).mean().item(), rate.item()
                if self.args.ssim:
                    psnr = mse
                else:
                    psnr = mse2psnr(mse)

                loss = self.args.lmda_scale * self.args.lmda * mse + rate

            ref_frame = rec_frame

            psnr_list.append(psnr)
            rate_list.append(rate)
            mse_list.append(mse)
            loss_list.append(loss)

        psnr = np.mean(psnr_list)
        rate = np.mean(rate_list)
        m_rate = np.mean(m_rate_list)
        mse = np.mean(mse_list)
        loss = np.mean(loss_list)

        logs = {'dataset_name': dataset_name, 'seq_name': seq_name, 
                'val_loss': loss, 'val_mse': mse, 
                'val_psnr': psnr, 'val_rate': rate, 
                'val_m_rate': m_rate}

        return {'val_log': logs}


    def validation_epoch_end(self, outputs):
        rd_dict = {}
        loss = []

        for logs in [log["val_log"] for log in outputs]:
            dataset_name = logs["dataset_name"]
            seq_name = logs["seq_name"]

            if not (dataset_name in rd_dict.keys()):
                rd_dict[dataset_name] = {}
                rd_dict[dataset_name]["psnr"] = []
                rd_dict[dataset_name]["rate"] = []
                rd_dict[dataset_name]["m_rate"] = []

            rd_dict[dataset_name]["psnr"].append(logs["val_psnr"].cpu().numpy())
            rd_dict[dataset_name]["rate"].append(logs["val_rate"].cpu().numpy())
            rd_dict[dataset_name]["m_rate"].append(logs["val_m_rate"].cpu().numpy())
            
            loss.append(logs["val_loss"].cpu().numpy())

        avg_loss = np.mean(loss)
        
        logs = {'val/loss': avg_loss}

        for dataset_name, rd in rd_dict.items():
            logs["val/'+dataset_name+' psnr"] = np.mean(rd["psnr"])
            logs["val/'+dataset_name+' rate"] = np.mean(rd["rate"])
            logs["val/'+dataset_name+' m_rate"] = np.mean(rd["m_rate"])

        self.log_dict(logs)

        return None

    def test_step(self, batch, batch_idx):
        metrics_name = ['PSNR', 'Rate', 'Mo_Rate', 'BDQ-PSNR', 'p1-PSNR', 'p1-BDQ-PSNR']
        metrics = {}
        for m in metrics_name:
            metrics[m] = []
        # PSNR: PSNR(gt, ADQ)
        # Rate
        # Mo_Rate: Motion Rate
        # BDQ-PSNR: PSNR(gt, BDQ)
        # p1-PSNR: PSNR(gt, ADQ) only when first P-frame in a GOP
        # p1-BDQ-PSNR: PSNR(gt, BDQ) only when first P-frame in a GOP

        dataset_name, seq_name, batch, frame_id_start = batch
        frame_id = int(frame_id_start)

        seq_name = seq_name[0]
        dataset_name = dataset_name[0]

        gop_size = batch.size(1)

        height, width = batch.size()[3:]

        psnr_list = []
        mc_psnr_list = []
        mse_list = []
        rate_list = []
        m_rate_list = []
        loss_list = []
        BDQ_psnr_list = []
        log_list = []

        align = Alignment()
        self.dpb = None

        os.makedirs(self.args.save_dir + f'/{seq_name}', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/flow', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/gt_frame', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/rec_frame', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/pred_frame', exist_ok=True)
        os.makedirs(self.args.save_dir + f'/{seq_name}/BDQ', exist_ok=True)

        for frame_idx in range(gop_size):
            TO_VISUALIZE = False and frame_id_start == 1 and frame_idx < 8 and seq_name in ["BasketballDrive', 'Kimono1', 'HoneyBee', 'Jockey"]

            if frame_idx != 0:
                coding_frame = batch[:, frame_idx]

                rec_frame, bpp, self.dpb, result = self(align.align(coding_frame), self.dpb, frame_idx)
                rec_frame = align.resume(rec_frame).clamp(0, 1)
                BDQ = align.resume(result["BDQ"]).clamp(0, 1)

                rate, mc_error = result["bpp"].item(), result["x2_mse"].item()
                m_rate = result["bpp_mv_y"].item() + result["bpp_mv_z"].item()

                mse = self.criterion(rec_frame, batch[:, frame_idx]).mean().item()
                if self.args.ssim:
                    psnr = mse
                else:
                    psnr = mse2psnr(mse)

                loss = self.args.lmda_scale * self.args.lmda * mse + \
                       rate + \
                       self.args.lmda_scale * self.args.lmda * 0.01 * mc_error

                if TO_VISUALIZE:
                    flow_map = plot_flow(data["flow_hat"])
                    save_image(flow_map,
                               self.args.save_dir + f'/{seq_name}/flow/'
                                                    f'frame_{int(frame_id_start + frame_idx)}_flow.png',
                               nrow=1)
                    save_image(coding_frame[0], self.args.save_dir + f'/{seq_name}/gt_frame/'
                                                                     f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(rec_frame[0], self.args.save_dir + f'/{seq_name}/rec_frame/'
                                                                  f'frame_{int(frame_id_start + frame_idx)}.png')
                    save_image(BDQ[0], self.args.save_dir + f'/{seq_name}/BDQ/'
                                                            f'frame_{int(frame_id_start + frame_idx)}.png')

                metrics["Mo_Rate"].append(m_rate)

                BDQ_psnr = mse2psnr(self.criterion(BDQ, coding_frame).mean().item())
                metrics["BDQ-PSNR"].append(BDQ_psnr)

                if frame_idx == 1:
                    metrics["p1-PSNR"].append(psnr)
                    metrics["p1-BDQ-PSNR"].append(BDQ_psnr)

                log_list.append({'PSNR': psnr, 'Rate': rate,
                                 'my': result["bpp_mv_y"].item(), 'mz': result["bpp_mv_z"].item(),
                                 'ry': result["bpp_y"].item(), 'rz': result["bpp_z"].item(),
                                 'BDQ-PSNR': BDQ_psnr})
            else:
                coding_frame = batch[:, frame_idx]
                rec_frame, rate, self.dpb, _ = self(align.align(coding_frame), None, frame_idx)
                rate = rate.item()

                rec_frame = align.resume(rec_frame).clamp(0, 1)

                save_image(coding_frame[0], self.args.save_dir + f'/{seq_name}/gt_frame/'
                                                                 f'frame_{int(frame_id_start + frame_idx)}.png')
                save_image(rec_frame[0], self.args.save_dir + f'/{seq_name}/rec_frame/'
                                                              f'frame_{int(frame_id_start + frame_idx)}.png')

                mse = self.criterion(rec_frame, coding_frame).mean().item()
                if self.args.ssim:
                    psnr = mse
                else:
                    psnr = mse2psnr(mse)

                loss = self.args.lmda_scale * self.args.lmda * mse + rate

                log_list.append({'PSNR': psnr, 'Rate': rate})

            metrics["PSNR"].append(psnr)
            metrics["Rate"].append(rate)

            frame_id += 1

        for m in metrics_name:
            metrics[m] = np.mean(metrics[m])

        logs = {'dataset_name': dataset_name, 'seq_name': seq_name, 'metrics': metrics, 'log_list': log_list,}

        return {'test_log': logs}


    def test_epoch_end(self, outputs):
        # dataset_name = {'HEVC-B': ['BasketballDrive', 'BQTerrace', 'Cactus', 'Kimono1', 'ParkScene'],
        #                 'UVG': ['Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ReadySteadyGo', 'ShakeNDry', 'YachtRide'],
        #                 'HEVC-C': ['BasketballDrill', 'BQMall', 'PartyScene', 'RaceHorses'],
        #                 'CLIC_2022': ['a06845dd7d1d808e4f4743b7f08f2bf75a9a72264d4fb16505caf6e334611003',
        #                             '57cb54c2cde2789359ecf11b9b9b8207c6a79b7aa27f15a69d7e9a1c2caad912',
        #                             'fae057c83b04868424da3bb7139e29b3f328d5a93aaa9e617e825b93422d92c5',
        #                             'af31d741db80475c531bb7182ad0536df9dc88a6876fa38386dd5db850d86051',
        #                             'd0a99fb6b64e60d7754265586481ec43968e8fd97e7e4437332bb182d7548cb3',
        #                             '97d6ac9d81b64bf909bf4898072bb20492522ae182918e763a86b56745890add',
        #                             'd73059fe0ed42169f7e98ff7401d00479a7969753eb80af9846176a42543ccb0',
        #                             '23e266612abe7b8767587d6e77a5eb3c6b8a71c6bf4c4ff2b1c11cc478cc7244',
        #                             '9a6379abea3fc820ca60afb9a60092d41b3a772ff348cfec92c062f6187f85e2',
        #                             '7c7d58e4f82772f627d5cbe3df3b08573d5bd7a58639387b865449d5a550bbda',
        #                             '29aabdd9d3065802c21e2d828561c205d563e79d39d1e10a18f961b5b5bf0cad',
        #                             '7b0eaacc48c9b5ea0edf5dcf352d913fd0cf3f79ae149e94ada89ba1e772e711',
        #                             '0442d8bdf9902226bfb38fbe039840d4f8ebe5270eda39d7dba56c2c3ae5becc',
        #                             'b7ee0264612a6ca6bf2bfa03df68acf4af9bb5cac34f7ad43fe30fa4b7bc4824',
        #                             '8db183688ce3e59461355e2c7cc97b3aee9f514a2e28260ead5a3ccf2000b079',
        #                             '8cbafab285e74614f10d3a8bf9ee94434eacae6332f5f10fe1e50bfe5de9ec33',
        #                             '318c694f5c83b78367da7e6584a95872510db8544f815120a86923aff00f5ff9',
        #                             '04ca8d2ac3af26ad4c5b14cf214e0d7c317c953e804810829d41645fdce1ad88',
        #                             '1e3224380c76fb4cad0a8d3a7c74a8d5bf0688d13df15f23acd2512de4374cb4',
        #                             '04a1274a93ec6a36ad2c1cb5eb83c3bdf2cf05bbe01c70a8ca846a7f9fa4b550',
        #                             '0d49152a92ce3b843968bf2e131ea5bc5e409ab056196e8c373f9bd2d31b303d',
        #                             '5d8f03cf5c6a469004a0ca73948ad64fa6d222b3b807f155a66684387f5d208a',
        #                             '0e1474478f33373566b4fbd6b357cf6b65015a6f4aa646754e065bf4a1b43c15',
        #                             '0659b03fb82cae130fef6a931755bbaae6e7bd88f58873df1ae98d2145dba9ce',
        #                             'a89f641b8dd2192f6f8b0ae75e3a24388b96023b21c63ff67bb359628f5df6de',
        #                             '209921b14cef20d62002e2b0c21ad692226135b52fee7eead315039ca51c470c',
        #                             '917d1b33f0e20d2d81471c3a0ff7adbef9e1fb7ee184b604880b280161ffdd56',
        #                             '9ce4af9a3b304b4b5387f27bca137ce1f0f35c12837c753fc17ea9bb49eb8ec5',
        #                             '393608bbbf2ac4d141ce6a3616a2364a2071539acb1969032012348c5817ef3c',
        #                             '9299df423938da4fd7f51736070420d2bb39d33972729b46a16180d07262df12']
        #                }

        metrics_name = list(outputs[0]["test_log"]["metrics"].keys())  # Get all metrics' names

        rd_dict = {}

        single_seq_logs = {}
        for metrics in metrics_name:
            single_seq_logs[metrics] = {}

        single_seq_logs["LOG"] = {}
        single_seq_logs["GOP"] = {}  # Will not be printed currently
        single_seq_logs["Seq_Names"] = []

        for logs in [log["test_log"] for log in outputs]:
            dataset_name = logs["dataset_name"]
            seq_name = logs["seq_name"]

            if not (dataset_name in rd_dict.keys()):
                rd_dict[dataset_name] = {}
                
                for metrics in metrics_name:
                    rd_dict[dataset_name][metrics] = []

            for metrics in logs["metrics"].keys():
                rd_dict[dataset_name][metrics].append(logs["metrics"][metrics])

            # Initialize
            if seq_name not in single_seq_logs["Seq_Names"]:
                single_seq_logs["Seq_Names"].append(seq_name)
                for metrics in metrics_name:
                    single_seq_logs[metrics][seq_name] = []
                single_seq_logs["LOG"][seq_name] = []
                single_seq_logs["GOP"][seq_name] = []

            # Collect metrics logs
            for metrics in metrics_name:
                single_seq_logs[metrics][seq_name].append(logs["metrics"][metrics])
            single_seq_logs["LOG"][seq_name].extend(logs["log_list"])
            single_seq_logs["GOP"][seq_name] = len(logs["log_list"])

        os.makedirs(self.args.save_dir + f'/report', exist_ok=True)

        for seq_name, log_list in single_seq_logs["LOG"].items():
            with open(self.args.save_dir + f'/report/{seq_name}.csv', 'w', newline='') as report:
                writer = csv.writer(report, delimiter=',')
                columns = ["frame"] + list(log_list[1].keys())
                writer.writerow(columns)

                # writer.writerow(["frame', 'PSNR', 'total bits', 'MC-PSNR', 'my', 'mz', 'ry', 'rz', 'MCerr-PSNR"])

                for idx in range(len(log_list)):
                    writer.writerow([f"frame_{idx + 1}"] + list(log_list[idx].values()))

        # Summary
        logs = {}
        print_log = '{:>16} '.format('Sequence_Name')
        for metrics in metrics_name:
            print_log += '{:>12}'.format(metrics)
        print_log += '\n'

        for seq_name in single_seq_logs["Seq_Names"]:
            print_log += '{:>16} '.format(seq_name[:5])

            for metrics in metrics_name:
                print_log += '{:12.4f}'.format(np.mean(single_seq_logs[metrics][seq_name]))

            print_log += '\n'
        print_log += '================================================\n'
        for dataset_name, rd in rd_dict.items():
            print_log += '{:>16} '.format(dataset_name)

            for metrics in metrics_name:
                logs["test/" + dataset_name + ' ' + metrics] = np.mean(rd[metrics])
                print_log += '{:12.4f}'.format(np.mean(rd[metrics]))

            print_log += '\n'

        print(print_log)

        with open(self.args.save_dir + f'/brief_summary.txt', 'w', newline='') as report:
            report.write(print_log)

        self.log_dict(logs)

        return None


    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        current_epoch = self.trainer.current_epoch
        
        lr_step = []
        for k, v in phase.items():
            if 'RNN' in k and v > current_epoch: 
                lr_step.append(v-current_epoch)
        lr_gamma = 0.5
        print('lr decay =', lr_gamma, 'lr milestones =', lr_step)

        optimizer = optim.AdamW([dict(params=self.main_parameters(), lr=self.args.lr)])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_step, lr_gamma)

        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=None,
                       using_native_amp=None, using_lbfgs=None):

        def clip_gradient(opt, grad_clip):
            for group in opt.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)

        clip_gradient(optimizer, 5)

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        #optimizer_closure()

    def compress(self, ref_frame, coding_frame, p_order):
        pass

    def decompress(self, ref_frame, strings, shapes, p_order):
        pass

    def setup(self, stage):

        self.logger.experiment.log_parameters(self.args)

        dataset_root = os.getenv('NEWDATAROOT')
        
        if stage == 'fit':
            transformer = transforms.Compose([
                transforms.RandomCrop((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

            self.train_dataset = VimeoDataset(dataset_root + "vimeo_septuplet/", 7, transform=transformer)
            self.val_dataset = VideoTestData(dataset_root + "video_dataset/", {85: 256, 170: 512, 380: 1024, 840: 2048}[self.args.lmda], first_gop=True)

        elif stage == 'test':
            self.test_dataset = VideoTestData(dataset_root + "video_dataset/", {85: 256, 170: 512, 380: 1024, 840: 2048}[self.args.lmda], sequence=('U', 'B'), GOP=self.args.test_GOP)

        else:
            raise NotImplementedError

    def train_dataloader(self):
        # REQUIRED
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.args.batch_size,
                                  num_workers=self.args.num_workers,
                                  shuffle=True)
        return train_loader

    def val_dataloader(self):
        # OPTIONAL
        val_loader = DataLoader(self.val_dataset,
                                batch_size=1,
                                num_workers=self.args.num_workers,
                                shuffle=False)
        return val_loader

    def test_dataloader(self):
        # OPTIONAL
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=1,
                                 num_workers=self.args.num_workers,
                                 shuffle=False)
        return test_loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the arguments for this LightningModule
        """
        # MODEL specific
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--no_feat_buffer', "-NFB", action="store_true")
        parser.add_argument('--learning_rate', '-lr', dest='lr', default=1e-4, type=float)
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--lmda', default=840, choices=[85, 170, 380, 840], type=int)
        parser.add_argument('--lmda_scale', default=1, type=float)
        parser.add_argument('--patch_size', default=256, type=int)
        parser.add_argument('--ssim', action="store_true")
        parser.add_argument('--debug', action="store_true")
        parser.add_argument('--inter_coder_type', default='CVAE', choices=['CVAE', 'CANF'], type=str)

        # training specific (for this model)
        parser.add_argument('--num_workers', default=16, type=int)
        parser.add_argument('--save_dir')

        return parser

if __name__ == '__main__':
    # sets seeds for numpy, torch, etc...
    # must do for DDP to work well

    #seed_everything(888888)
    #torch.backends.cudnn.deterministic = True

    save_root = os.getenv('LOG', './') + '/torchDVC/'

    parser = argparse.ArgumentParser(add_help=True)

    # add args from trainer
    parser = Trainer.add_argparse_args(parser)

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = Pframe.add_model_specific_args(parser)

    # training specific
    parser.add_argument('--restore', type=str, choices=['none', 'resume', 'load', 'custom', 'finetune'], default='none')
    parser.add_argument('--restore_exp_key', type=str, default=None)
    parser.add_argument('--restore_exp_epoch', type=int, default=0)
    parser.add_argument('--test', "-T", action="store_true")
    parser.add_argument('--test_GOP', type=int, default=32)
    parser.add_argument('--experiment_name', type=str, default='basic')
    parser.add_argument('--project_name', type=str, default="CANFVC_DMC")

    # parse params
    args = parser.parse_args()

    experiment_name = args.experiment_name
    project_name = args.project_name

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        save_last=True,
        every_n_epochs=1, # Save at least every 10 epochs
        verbose=True,
        monitor='val/loss',
        mode='min',
        filename = '{epoch}'
    )

    db = None
    if int(args.devices) > 1:
        #db = DDPStrategy(process_group_backend="gloo")
        db = "dp"

    comet_logger = CometLogger(
        api_key="bFaTNhLcuqjt1mavz02XPVwN8",
        project_name=project_name,
        workspace="tl32rodan",
        experiment_name=experiment_name + "-" + str(args.lmda),
        experiment_key = args.restore_exp_key if args.restore == 'resume' else None,
        disabled=args.test or args.debug
    )
    
    args.save_dir = os.path.join(save_root, project_name, experiment_name + '-' + str(args.lmda))
    
    if args.restore == 'resume' or args.restore == 'finetune':
        trainer = Trainer.from_argparse_args(args,
                                             enable_checkpointing=True,
                                             callbacks=checkpoint_callback,
                                             accelerator='gpu',
                                             strategy=db,
                                             logger=comet_logger,
                                             default_root_dir=save_root,
                                             check_val_every_n_epoch=1,
                                             num_sanity_val_steps=0,
                                             log_every_n_steps=50,
                                             detect_anomaly=True,
                                             limit_train_batches=0.25,
                                             max_epochs=-1
                                            )

        epoch_num = args.restore_exp_epoch
        if args.restore_exp_key is None:
            raise ValueError
        else:  # When prev_exp_key is specified in args
            checkpoint = torch.load(os.path.join(save_root, project_name, args.restore_exp_key, "checkpoints", f"epoch={epoch_num}.ckpt"),
                                    map_location=(lambda storage, loc: storage))

        #if args.restore == 'resume':
        #    trainer.current_epoch = epoch_num + 1
        #else:
        #    trainer.current_epoch = phase["trainAll_2frames"]

        model = Pframe(args)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    
    elif args.restore == 'load':
        trainer = Trainer.from_argparse_args(args,
                                             enable_checkpointing=True,
                                             callbacks=checkpoint_callback,
                                             accelerator='gpu',
                                             strategy=db,
                                             logger=comet_logger,
                                             default_root_dir=save_root,
                                             check_val_every_n_epoch=1,
                                             num_sanity_val_steps=0,
                                             log_every_n_steps=50,
                                             detect_anomaly=True,
                                             limit_train_batches=0.15,
                                             max_epochs=-1
                                            )
        
        epoch_num = args.restore_exp_epoch
        if args.restore_exp_key is None:
            raise ValueError
        else:  # When prev_exp_key is specified in args
            checkpoint = torch.load(os.path.join(save_root, project_name, args.restore_exp_key, "checkpoints",
                                                 f"epoch={epoch_num}.ckpt"),
                                    map_location=(lambda storage, loc: storage))

        #trainer.current_epoch = phase["trainAll_fullgop"] - 2
        #trainer.current_epoch = epoch_num + 1
        #trainer.current_epoch = phase["train_aux"] 


        model = Pframe(args)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        #summary(model.Residual.DQ)
    
    elif args.restore == 'custom':
        trainer = Trainer.from_argparse_args(args,
                                             enable_checkpointing=True,
                                             callbacks=checkpoint_callback,
                                             accelerator='gpu',
                                             strategy=db,
                                             logger=comet_logger,
                                             default_root_dir=save_root,
                                             check_val_every_n_epoch=1,
                                             num_sanity_val_steps=0,
                                             log_every_n_steps=50,
                                             detect_anomaly=True,
                                             limit_train_batches=0.25,
                                             max_epochs=-1
                                            )
        
        from src.models.video_model import DMC
        i_frame_q_scales = IntraNoAR.get_q_scales_from_ckpt('./checkpoints/acmmm2022_image_psnr.pth.tar')
        i_state_dict = get_state_dict('./checkpoints/acmmm2022_image_psnr.pth.tar')
        p_frame_y_q_scales, p_frame_mv_y_q_scales = DMC.get_q_scales_from_ckpt('./checkpoints/acmmm2022_video_psnr.pth.tar')
        p_state_dict = get_state_dict('./checkpoints/acmmm2022_video_psnr.pth.tar')

        from collections import OrderedDict
        new_ckpt = OrderedDict()

        for k, v in i_state_dict.items():
            key = 'i_model.' + k
            new_ckpt[key] = v
        for k, v in p_state_dict.items():
            if k.split('.')[0] == 'contextual_encoder':
                key = '.'.join(['p_model', 'contextual_coder', 'analysis0', 'model'] + k.split('.')[1:])
                new_ckpt[key] = v
                key = '.'.join(['p_model', 'contextual_coder', 'analysis1', 'model'] + k.split('.')[1:])
                new_ckpt[key] = v
            elif k.split('.')[0] == 'contextual_decoder':
                key = '.'.join(['p_model', 'contextual_coder', 'synthesis0', 'model_part1'] + k.split('.')[1:])
                new_ckpt[key] = v
                key = '.'.join(['p_model', 'contextual_coder', 'synthesis1', 'model_part1'] + k.split('.')[1:])
                new_ckpt[key] = v
            elif k.split('.')[0] == 'recon_generation_net':
                key = '.'.join(['p_model', 'contextual_coder', 'synthesis0', 'model_part2'] + k.split('.')[1:])
                new_ckpt[key] = v
                key = '.'.join(['p_model', 'contextual_coder', 'synthesis1', 'model_part2'] + k.split('.')[1:])
                new_ckpt[key] = v
            else:
                key = 'p_model.' + k
                new_ckpt[key] = v

        model = Pframe(args)
        model.load_state_dict(new_ckpt, strict=True)

    else:
        trainer = Trainer.from_argparse_args(args,
                                             enable_checkpointing=True,
                                             callbacks=checkpoint_callback,
                                             accelerator='gpu',
                                             strategy=db,
                                             logger=comet_logger,
                                             default_root_dir=save_root,
                                             check_val_every_n_epoch=1,
                                             num_sanity_val_steps=0,
                                             log_every_n_steps=50,
                                             detect_anomaly=True,
                                             limit_train_batches=0.1,
                                             max_epochs=-1
                                            )
     
        from src.models.video_model import DMC
        i_frame_q_scales = IntraNoAR.get_q_scales_from_ckpt('./checkpoints/acmmm2022_image_psnr.pth.tar')
        i_state_dict = get_state_dict('./checkpoints/acmmm2022_image_psnr.pth.tar')
        p_frame_y_q_scales, p_frame_mv_y_q_scales = DMC.get_q_scales_from_ckpt('./checkpoints/acmmm2022_video_psnr.pth.tar')
        p_state_dict = get_state_dict('./checkpoints/acmmm2022_video_psnr.pth.tar')

        from collections import OrderedDict
        new_ckpt = OrderedDict()

        for k, v in i_state_dict.items():
            key = 'i_model.' + k
            new_ckpt[key] = v
        for k, v in p_state_dict.items():
            key = 'p_model.' + k
            new_ckpt[key] = v

        new_ckpt['i_model.q_scale'] = i_frame_q_scales.view(4, 1, 1, 1)
        new_ckpt['p_model.y_q_scale'] = p_frame_y_q_scales.view(4, 1, 1, 1)
        new_ckpt['p_model.mv_y_q_scale'] = p_frame_mv_y_q_scales.view(4, 1, 1, 1)

        model = Pframe(args)
        model.load_state_dict(new_ckpt, strict=True)

        #summary(model.Motion)
        #summary(model.CondMotion)
        #summary(model.Residual)
        #summary(model)

    if args.test:
        trainer.test(model)
    else:
        trainer.fit(model)
