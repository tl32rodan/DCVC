import torch
import random
import os
from glob import glob
import subprocess
import numpy as np

from torch import stack
from torch.utils.data import Dataset as torchData
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import default_loader as imgloader
from subprocess import Popen, PIPE

from PIL import Image


rgb_transform = transforms.Compose([
    transforms.ToTensor()])

class VimeoDataset(torchData):
    """ vimeo_septuplet dataset
        download: `$ wget http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip`

    Args:
        root
        mode
        frames
        transform
    """

    def __init__(self, root, frames=7, transform=rgb_transform):
        super().__init__()
        self.folder = glob(os.path.join(root, 'sequences/*/*/'))
        self.frames = frames
        self.transform = transform

    def __len__(self):
        return len(self.folder)

    @property
    def info(self):
        gop = self[0]
        return "\nGop size: {}".format(gop.shape)

    def __getitem__(self, index):
        path = self.folder[index]
        seed = random.randint(0, 1e9)
        imgs = []
        for f in range(self.frames):
            random.seed(seed)
            torch.manual_seed(seed)
            raw_path = os.path.join(path, 'im' + str(f + 1) + '.png')
            imgs.append(self.transform(imgloader(raw_path)))

        return stack(imgs)


class VideoTestSequence(torchData):
    def __init__(self, root, lmda, dataset='U', sequence='Beauty', seq_len=600, GOP=12):
        super(VideoTestSequence, self).__init__()

        self.root = root
        self.lmda = lmda
        self.qp = {256: 37, 512: 32, 1024: 27, 2048: 22, 4096: 22}[lmda]

        assert os.path.exists(os.path.join(self.root, dataset, sequence)), \
            f'FileNotFoundError: sequence is not found (path: {os.path.join(self.root, dataset, sequence)})'
        
        if not (seq_len % seq_len):
            print(f'Warning: GOP={GOP} cannot divide seq_len={seq_len} ; Only the first {seq_len - seq_len % GOP} will be coded')

        self.seq_name = [sequence]
        seq_len = [seq_len - seq_len % GOP]
        gop_size = [GOP]
        dataset_name_list = [dataset]

        seq_len = dict(zip(self.seq_name, seq_len))
        gop_size = dict(zip(self.seq_name, gop_size))
        dataset_name_list = dict(zip(self.seq_name, dataset_name_list))

        self.gop_list = []

        for seq_name in self.seq_name:
            for gop_idx in range(seq_len[seq_name] // gop_size[seq_name]):
                self.gop_list.append([dataset_name_list[seq_name],
                                      seq_name,
                                      1 + gop_size[seq_name] * gop_idx,
                                      1 + gop_size[seq_name] * (gop_idx + 1)])
        
    def __len__(self):
        return len(self.gop_list)

    def __getitem__(self, idx):
        dataset_name, seq_name, frame_start, frame_end = self.gop_list[idx]
        seed = random.randint(0, 1e9)
        imgs = []
         
        # First image of `imgs` will be BPG-compressed frame
        for frame_idx in range(frame_start, frame_end):
            random.seed(seed)

            raw_path = os.path.join(self.root, dataset_name, seq_name, 'frame_{:d}.png'.format(frame_idx))

            imgs.append(transforms.ToTensor()(imgloader(raw_path)))

        return dataset_name, seq_name, stack(imgs), frame_start


class VideoTestData(torchData):
    def __init__(self, root, lmda, first_gop=False, sequence=('U', 'B'), GOP=12):
        super(VideoTestData, self).__init__()
        
        assert GOP in [12, 16, 32], ValueError
        self.root = root
        self.lmda = lmda
        self.qp = {256: 37, 512: 32, 1024: 27, 2048: 22, 4096: 22}[lmda]

        self.seq_name = []
        seq_len = []
        gop_size = []
        dataset_name_list = []

        if 'U' in sequence:
            self.seq_name.extend(['Beauty', 'Bosphorus', 'HoneyBee', 'Jockey', 'ReadySteadyGo', 'ShakeNDry', 'YachtRide'])
            if GOP in [12, 16]:
                seq_len.extend([600, 600, 600, 600, 600, 300, 600])
            else:
                seq_len.extend([96]*7)
            gop_size.extend([GOP]*7)
            dataset_name_list.extend(['U']*7)
        if 'B' in sequence:
            self.seq_name.extend(['Kimono1', 'BQTerrace', 'Cactus', 'BasketballDrive', 'ParkScene'])
            if GOP in [12, 16]:
                seq_len.extend([100]*5)
                #seq_len.extend([240, 600, 500, 500, 240])
            else:
                seq_len.extend([96]*5)
            gop_size.extend([GOP]*5)
            dataset_name_list.extend(['B']*5)
        if 'C' in sequence:
            self.seq_name.extend(['BasketballDrill', 'BQMall', 'PartyScene', 'RaceHorses'])
            if GOP in [12, 16]:
                seq_len.extend([100]*4)
            else:
                seq_len.extend([96]*4)
            gop_size.extend([GOP]*4)
            dataset_name_list.extend(['C']*4)
        if 'D' in sequence:
            self.seq_name.extend(['BasketballPass', 'BQSquare', 'BlowingBubbles', 'RaceHorses1']) # Rename "RaceHorses" to avoid repetition
            if GOP in [12, 16]:
                seq_len.extend([100]*4)
            else:
                seq_len.extend([96]*4)
            gop_size.extend([GOP]*4)
            dataset_name_list.extend(['D']*4)
        if 'E' in sequence:
            self.seq_name.extend(['vidyo1', 'vidyo3', 'vidyo4'])
            if GOP in [12, 16]:
                seq_len.extend([100]*3)
            else:
                seq_len.extend([96]*3)
            gop_size.extend([GOP]*3)
            dataset_name_list.extend(['E']*3)

        if 'M' in sequence:
            MCL_list = []
            for i in range(1, 31):
                MCL_list.append('videoSRC'+str(i).zfill(2))
                
            self.seq_name.extend(MCL_list)
            if GOP in [12, 16]:
                seq_len.extend([150, 150, 150, 150, 125, 125, 125, 125, 125, 150,
                                150, 150, 150, 150, 150, 150, 120, 125, 150, 125,
                                120, 120, 120, 120, 120, 150, 150, 150, 120, 150])
            else:
                seq_len.extend([96]*30)
            gop_size.extend([GOP]*30)
            dataset_name_list.extend(['M']*30)

        seq_len = dict(zip(self.seq_name, seq_len))
        gop_size = dict(zip(self.seq_name, gop_size))
        dataset_name_list = dict(zip(self.seq_name, dataset_name_list))

        self.gop_list = []

        for seq_name in self.seq_name:
            if first_gop:
                gop_num = 1
            else:
                gop_num = seq_len[seq_name] // gop_size[seq_name]
            for gop_idx in range(gop_num):
                self.gop_list.append([dataset_name_list[seq_name],
                                      seq_name,
                                      1 + gop_size[seq_name] * gop_idx,
                                      1 + gop_size[seq_name] * (gop_idx + 1)])
        
    def __len__(self):
        return len(self.gop_list)

    def __getitem__(self, idx):
        dataset_name, seq_name, frame_start, frame_end = self.gop_list[idx]
        seed = random.randint(0, 1e9)
        imgs = []
         
        for frame_idx in range(frame_start, frame_end):
            random.seed(seed)

            raw_path = os.path.join(self.root, dataset_name, seq_name, 'frame_{:d}.png'.format(frame_idx))
            
            imgs.append(transforms.ToTensor()(imgloader(raw_path)))

        return dataset_name, seq_name, stack(imgs), frame_start


class BitstreamData(VideoTestData):
    def __init__(self, root, lmda, sequence=('U', 'B'), GOP=12, bin_root='./bin', load_Iframe=False):
        super(BitstreamData, self).__init__(root, lmda, sequence, GOP)
        self.bin_root = bin_root
        self.load_Iframe = load_Iframe # Load BPG I-frame when True

    def __getitem__(self, idx):
        dataset_name, seq_name, frame_start, frame_end = self.gop_list[idx]
        filenames = []
         
        for frame_idx in range(frame_start, frame_end):

            filename = os.path.join(self.bin_root, dataset_name, seq_name, f'{frame_idx}.bin')

            if frame_idx == frame_start and self.load_Iframe:
                img_root = os.path.join(self.root, 'bpg', str(self.qp), 'decoded', seq_name)
                os.makedirs(img_root, exist_ok=True)

                img_path = os.path.join(img_root, f'frame_{frame_idx}.png')

                if not os.path.exists(img_path):
                    # Compress data on-the-fly when they are not previously compressed.
                    bin_path = img_path.replace('decoded', 'bin').replace('png', 'bin')

                    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)

                    subprocess.call(f'{libbpg_path}/bpgenc -f 444 -q {self.qp} -o {bin_path} {raw_path}'.split(' '))
                    subprocess.call(f'{libbpg_path}/bpgdec -o {img_path} {bin_path}'.split(' '))

                filenames.append(transforms.ToTensor()(imgloader(img_path)))

            else:
                filenames.append(filename)

        return dataset_name, seq_name, filenames, frame_start


class BitstreamSequence(VideoTestSequence):
    def __init__(self, root, lmda, dataset='U', sequence='Beauty', seq_len=600, GOP=12, bin_root='./bin', load_Iframe=False):
        super(BitstreamSequence, self).__init__(root, lmda, dataset, sequence, seq_len, GOP)
        self.bin_root = bin_root
        self.load_Iframe = load_Iframe # Load BPG I-frame when True

    def __getitem__(self, idx):
        dataset_name, seq_name, frame_start, frame_end = self.gop_list[idx]
        filenames = []
         
        for frame_idx in range(frame_start, frame_end):

            filename = os.path.join(self.bin_root, dataset_name, seq_name, f'{frame_idx}.bin')

            if frame_idx == frame_start and self.load_Iframe:
                img_root = os.path.join(self.root, 'bpg', str(self.qp), 'decoded', seq_name)
                os.makedirs(img_root, exist_ok=True)

                img_path = os.path.join(img_root, f'frame_{frame_idx}.png')

                if not os.path.exists(img_path):
                    # Compress data on-the-fly when they are not previously compressed.
                    bin_path = img_path.replace('decoded', 'bin').replace('png', 'bin')

                    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)

                    subprocess.call(f'{libbpg_path}/bpgenc -f 444 -q {self.qp} -o {bin_path} {raw_path}'.split(' '))
                    subprocess.call(f'{libbpg_path}/bpgdec -o {img_path} {bin_path}'.split(' '))

                filenames.append(transforms.ToTensor()(imgloader(img_path)))

            else:
                filenames.append(filename)

        return dataset_name, seq_name, filenames, frame_start
