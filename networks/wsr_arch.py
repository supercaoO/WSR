import numpy as np
from networks.block import *
from pytorch_wavelets import DWTInverse


class RecurrentBlock(nn.Module):
    def __init__(self, num_features, num_simdb, act_type, norm_type):
        super(RecurrentBlock, self).__init__()
        self.compress_in = ConvBlock(2 * num_features, num_features, kernel_size=1, act_type=act_type,
                                     norm_type=norm_type)
        self.SIMDBs = []
        for _ in range(num_simdb):
            self.SIMDBs.append(SIMDB(in_channels=num_features))
        self.SIMDBs = nn.Sequential(*self.SIMDBs)

        self.should_reset = True
        self.last_hidden = None

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).to(x.device)
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)
        x = self.compress_in(x)
        out = self.SIMDBs(x)
        self.last_hidden = out
        return out

    def reset_state(self):
        self.should_reset = True


class WSR(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_simdb, upscale_factor, act_type='prelu',
                 norm_type=None):
        super(WSR, self).__init__()

        padding = 2
        self.num_features = num_features
        self.upscale_factor = upscale_factor
        self.num_steps = int(np.log2(self.upscale_factor) + 1)

        # LR feature extraction block
        self.conv_in = ConvBlock(in_channels, 4 * num_features,
                                 kernel_size=3,
                                 act_type=act_type, norm_type=norm_type)
        self.feat_in = ConvBlock(4 * num_features, num_features,
                                 kernel_size=1,
                                 act_type=act_type, norm_type=norm_type)

        # recurrent block
        self.rb = RecurrentBlock(num_features, num_simdb, act_type, norm_type)

        # reconstruction block
        self.conv_steps = nn.ModuleList([
            nn.Sequential(ConvBlock(num_features, num_features, kernel_size=3, act_type=act_type, norm_type=norm_type),
                          ConvBlock(num_features, out_channels, kernel_size=3, act_type=None, norm_type=norm_type)),

            nn.Sequential(ConvBlock(num_features, num_features, kernel_size=3, act_type=act_type, norm_type=norm_type),
                          ConvBlock(num_features, out_channels * 3, kernel_size=3, act_type=None, norm_type=norm_type))]
        )
        for step in range(2, self.num_steps):
            conv_step = nn.Sequential(
                DeconvBlock(num_features, num_features, kernel_size=int(2 ** (step - 1) + 4),
                            stride=int(2 ** (step - 1)), padding=padding, act_type=act_type, norm_type=norm_type),
                ConvBlock(num_features, out_channels * 3, kernel_size=3, act_type=None, norm_type=norm_type))
            self.conv_steps.append(conv_step)

        # inverse wavelet transformation
        self.ifm = DWTInverse(wave='db1', mode='symmetric').eval()
        for k, v in self.ifm.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        self._reset_state()

        x = self.conv_in(x)
        x = self.feat_in(x)

        Yl = self.conv_steps[0](x)

        Yh = []
        for step in range(1, self.num_steps):
            h = self.rb(x)
            h = self.conv_steps[step](h)
            h = h.view(h.size()[0], h.size()[1] // 3, 3, h.size()[2], h.size()[3])
            Yh = [h] + Yh

        sr = self.ifm((Yl, Yh))

        return [Yl, Yh, sr]

    def _reset_state(self):
        self.rb.reset_state()
