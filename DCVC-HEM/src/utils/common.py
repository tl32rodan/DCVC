# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import os
from unittest.mock import patch

import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

    raise argparse.ArgumentTypeError('Boolean value expected.')


def interpolate_log(min_val, max_val, num, decending=True):
    assert max_val > min_val
    assert min_val > 0
    if decending:
        values = np.linspace(np.log(max_val), np.log(min_val), num)
    else:
        values = np.linspace(np.log(min_val), np.log(max_val), num)
    values = np.exp(values)
    return values


def scale_list_to_str(scales):
    s = ''
    for scale in scales:
        s += f'{scale:.2f} '

    return s


def create_folder(path, print_if_create=False):
    if not os.path.exists(path):
        os.makedirs(path)
        if print_if_create:
            print(f"created folder: {path}")


@patch('json.encoder.c_make_encoder', None)
def dump_json(obj, fid, float_digits=-1, **kwargs):
    of = json.encoder._make_iterencode  # pylint: disable=W0212

    def inner(*args, **kwargs):
        args = list(args)
        # fifth argument is float formater which we will replace
        args[4] = lambda o: format(o, '.%df' % float_digits)
        return of(*args, **kwargs)

    with patch('json.encoder._make_iterencode', wraps=inner):
        json.dump(obj, fid, **kwargs)


def generate_log_json(frame_num, frame_types, bits, psnrs, ssims,
                      frame_pixel_num, test_time):
    cur_ave_i_frame_bit = 0
    cur_ave_i_frame_psnr = 0
    cur_ave_i_frame_msssim = 0
    cur_ave_p_frame_bit = 0
    cur_ave_p_frame_psnr = 0
    cur_ave_p_frame_msssim = 0
    cur_i_frame_num = 0
    cur_p_frame_num = 0
    for idx in range(frame_num):
        if frame_types[idx] == 0:
            cur_ave_i_frame_bit += bits[idx]
            cur_ave_i_frame_psnr += psnrs[idx]
            cur_ave_i_frame_msssim += ssims[idx]
            cur_i_frame_num += 1
        else:
            cur_ave_p_frame_bit += bits[idx]
            cur_ave_p_frame_psnr += psnrs[idx]
            cur_ave_p_frame_msssim += ssims[idx]
            cur_p_frame_num += 1

    log_result = {}
    log_result['frame_pixel_num'] = frame_pixel_num
    log_result['i_frame_num'] = cur_i_frame_num
    log_result['p_frame_num'] = cur_p_frame_num
    log_result['ave_i_frame_bpp'] = cur_ave_i_frame_bit / cur_i_frame_num / frame_pixel_num
    log_result['ave_i_frame_psnr'] = cur_ave_i_frame_psnr / cur_i_frame_num
    log_result['ave_i_frame_msssim'] = cur_ave_i_frame_msssim / cur_i_frame_num
    log_result['frame_bpp'] = list(np.array(bits) / frame_pixel_num)
    log_result['frame_psnr'] = psnrs
    log_result['frame_msssim'] = ssims
    log_result['frame_type'] = frame_types
    log_result['test_time'] = test_time
    if cur_p_frame_num > 0:
        total_p_pixel_num = cur_p_frame_num * frame_pixel_num
        log_result['ave_p_frame_bpp'] = cur_ave_p_frame_bit / total_p_pixel_num
        log_result['ave_p_frame_psnr'] = cur_ave_p_frame_psnr / cur_p_frame_num
        log_result['ave_p_frame_msssim'] = cur_ave_p_frame_msssim / cur_p_frame_num
    else:
        log_result['ave_p_frame_bpp'] = 0
        log_result['ave_p_frame_psnr'] = 0
        log_result['ave_p_frame_msssim'] = 0
    log_result['ave_all_frame_bpp'] = (cur_ave_i_frame_bit + cur_ave_p_frame_bit) / \
        (frame_num * frame_pixel_num)
    log_result['ave_all_frame_psnr'] = (cur_ave_i_frame_psnr + cur_ave_p_frame_psnr) / frame_num
    log_result['ave_all_frame_msssim'] = (cur_ave_i_frame_msssim + cur_ave_p_frame_msssim) / \
        frame_num

    return log_result


class Alignment(torch.nn.Module):
    """Image Alignment for model downsample requirement"""

    def __init__(self, divisor=64., mode='pad', padding_mode='replicate'):
        super().__init__()
        self.divisor = float(divisor)
        self.mode = mode
        self.padding_mode = padding_mode
        self._tmp_shape = None

    def extra_repr(self):
        s = 'divisor={divisor}, mode={mode}'
        if self.mode == 'pad':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    @staticmethod
    def _resize(input, size):
        return F.interpolate(input, size, mode='bilinear', align_corners=False)

    def _align(self, input):
        H, W = input.size()[-2:]
        H_ = int(ceil(H / self.divisor) * self.divisor)
        W_ = int(ceil(W / self.divisor) * self.divisor)
        pad_H, pad_W = H_-H, W_-W
        if pad_H == pad_W == 0:
            self._tmp_shape = None
            return input

        self._tmp_shape = input.size()
        if self.mode == 'pad':
            return F.pad(input, (0, pad_W, 0, pad_H), mode=self.padding_mode)
        elif self.mode == 'resize':
            return self._resize(input, size=(H_, W_))

    def _resume(self, input, shape=None):
        if shape is not None:
            self._tmp_shape = shape
        if self._tmp_shape is None:
            return input

        if self.mode == 'pad':
            output = input[..., :self._tmp_shape[-2], :self._tmp_shape[-1]]
        elif self.mode == 'resize':
            output = self._resize(input, size=self._tmp_shape[-2:])

        return output

    def align(self, input):
        """align"""
        if input.dim() == 4:
            return self._align(input)
        elif input.dim() == 5:
            return split_k(self._align(cat_k(input)), input.size(0))

    def resume(self, input, shape=None):
        """resume"""
        if input.dim() == 4:
            return self._resume(input, shape)
        elif input.dim() == 5:
            return split_k(self._resume(cat_k(input), shape), input.size(0))

    def forward(self, func, *args, **kwargs):
        pass


