# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch import nn


def conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3(in_ch, out_ch, r=1):
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


def subpel_conv1x1(in_ch, out_ch, r=1):
    """1x1 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )


def conv1x1(in_ch, out_ch, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU()
        self.conv2 = conv3x3(out_ch, out_ch)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        if stride != 1:
            self.downsample = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch, out_ch, upsample=2):
        super().__init__()
        self.subpel_conv = subpel_conv1x1(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU()
        self.conv = conv3x3(out_ch, out_ch)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.upsample = subpel_conv1x1(in_ch, out_ch, upsample)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.leaky_relu2(out)
        identity = self.upsample(x)
        out += identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch, out_ch, leaky_relu_slope=0.01):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.conv2 = conv3x3(out_ch, out_ch)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        out = out + identity
        return out


class AugmentedNormalizedFlow(nn.Sequential):

    def __init__(self, *args, use_affine, transpose, distribution='gaussian'):
        super(AugmentedNormalizedFlow, self).__init__(*args)
        self.use_affine = use_affine
        self.transpose = transpose
        self.distribution = distribution
        if distribution == 'gaussian':
            self.init_code = torch.randn_like
        elif distribution == 'uniform':
            self.init_code = torch.rand_like
        elif distribution == 'zeros':
            self.init_code = torch.zeros_like

    def get_condition(self, input, jac=None):
        condition = super().forward(input)
        if self.use_affine:
            loc, scale = condition.chunk(2, dim=1)
        else:
            loc, scale = condition, input.new_zeros(input.size(0), 1)

        self.scale = lower_bound(scale, 0.11)
        self.jacobian(jac)

        scale = self.scale
        if self.use_affine:
            condition = torch.cat([loc, scale], dim=1)
        else:
            condition = loc
        return condition, jac

    def get_condition2(self, input, jac=None):
        condition = super().forward(input)
        if self.use_affine:
            loc, scale = condition.chunk(2, dim=1)
        else:
            loc, scale = input.new_zeros(input.size(0), 1), condition

        self.scale = lower_bound(scale, 0.11)
        self.jacobian(jac)

        scale = self.scale
        if self.use_affine:
            condition = torch.cat([loc, scale], dim=1)
        else:
            condition = scale
        return condition, jac

    def forward(self, input, code=None, jac=None, rev=False, last_layer=False):
        if self.transpose:
            input, code = code, input

        condition = super().forward(input)
        if self.use_affine:
            loc, scale = condition.chunk(2, dim=1)
        else:
            loc, scale = condition, input.new_zeros(input.size(0), 1)

        self.scale = scale.sigmoid() * 2 - 1
        scale = self.scale

        if code is None:
            if self.use_affine:
                code = self.init_code(loc)
            else:
                code = None

        if (not rev) ^ self.transpose:
            if code is None:

                code = loc
            else:
                if self.use_affine and not last_layer:
                    code = code * scale.exp()
                    self.jacobian(jac, rev=rev)

                code = code + loc

        else:
            code = code - loc

            if self.use_affine and not last_layer:
                code = code / scale.exp()
                self.jacobian(jac, rev=rev)

        if self.transpose:
            input, code = code, input
        return input, code, jac

    def jacobian(self, jacs=None, rev=False):
        if jacs is not None:
            jac = self.scale.flatten(1).sum(1)
            if rev ^ self.transpose:
                jac = jac * -1

            jacs.append(jac)
        else:
            jac = None
        return jac


class CondAugmentedNormalizedFlow(nn.Module):
    def __init__(self, use_affine, transpose, distribution='gaussian'):
        super(CondAugmentedNormalizedFlow, self).__init__()
        self.use_affine = use_affine
        self.transpose = transpose
        self.distribution = distribution
        if distribution == 'gaussian':
            self.init_code = torch.randn_like
        elif distribution == 'uniform':
            self.init_code = torch.rand_like
        elif distribution == 'zeros':
            self.init_code = torch.zeros_like

    def get_condition(self, input, jac=None):
        condition = super().forward(input)
        if self.use_affine:
            loc, scale = condition.chunk(2, dim=1)
        else:
            loc, scale = condition, input.new_zeros(input.size(0), 1)

        self.scale = lower_bound(scale, 0.11)
        self.jacobian(jac)

        scale = self.scale
        if self.use_affine:
            condition = torch.cat([loc, scale], dim=1)
        else:
            condition = loc
        return condition, jac

    def get_condition2(self, input, jac=None):
        condition = super().forward(input)
        if self.use_affine:
            loc, scale = condition.chunk(2, dim=1)
        else:
            loc, scale = input.new_zeros(input.size(0), 1), condition

        self.scale = lower_bound(scale, 0.11)
        self.jacobian(jac)

        scale = self.scale
        if self.use_affine:
            condition = torch.cat([loc, scale], dim=1)
        else:
            condition = scale
        return condition, jac

    def forward(self, input, input_cond, code=None, jac=None, rev=False, last_layer=False):
        if self.transpose:
            input, code = code, input

        condition = self.net_forward(input, input_cond)

        if self.use_affine:
            loc, scale = condition.chunk(2, dim=1)
        else:
            loc, scale = condition, input.new_zeros(input.size(0), 1)

        self.scale = scale.sigmoid() * 2 - 1
        scale = self.scale

        if code is None:
            if self.use_affine:
                code = self.init_code(loc)
            else:
                code = None

        if (not rev) ^ self.transpose:
            if code is None:

                code = loc
            else:
                if self.use_affine and not last_layer:
                    code = code * scale.exp()
                    self.jacobian(jac, rev=rev)

                code = code + loc

        else:
            code = code - loc

            if self.use_affine and not last_layer:
                code = code / scale.exp()
                self.jacobian(jac, rev=rev)

        if self.transpose:
            input, code = code, input
        return input, code, jac

    def jacobian(self, jacs=None, rev=False):
        if jacs is not None:
            jac = self.scale.flatten(1).sum(1)
            if rev ^ self.transpose:
                jac = jac * -1

            jacs.append(jac)
        else:
            jac = None
        return jac

    def net_forward(self, input, input_cond):

        raise NotImplementedError
 
