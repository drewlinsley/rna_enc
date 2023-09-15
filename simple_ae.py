from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F



class Conv2dBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = norm_layer(out_channels)
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Unet(nn.Module):
    def __init__(
            self,
            in_chans=1,
            num_classes=5,
            encs=3,
            decs=3,
            kernel_size=3,
            int_chans=128,
            latent_dim=4,
            norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        enc_blocks = []
        in_ch = in_chans
        out_ch = int_chans
        for _ in range(encs):
            # enc_blocks.append(Conv2dBnAct(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size))
            enc_blocks.append(
                nn.Sequential(
                    *[
                        nn.Conv2d(
                            in_ch,
                            out_ch,
                            kernel_size,
                            stride=1,
                            padding=kernel_size // 2),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                    ]
                )
            )
            in_ch = out_ch
        enc_blocks.append(
            nn.Sequential(
                *[
                    nn.Conv2d(
                        in_ch,
                        latent_dim,
                        kernel_size,
                        stride=1,
                        padding=kernel_size // 2),
                    # nn.BatchNorm2d(out_ch),
                    # nn.ReLU(inplace=True),
                ]
            )
        )
        self.encoder_blocks = nn.Sequential(*enc_blocks)

        dec_blocks = []
        in_ch = latent_dim
        out_ch = int_chans
        for _ in range(encs):
            # dec_blocks.append(Conv2dBnAct(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size))
            dec_blocks.append(
                nn.Sequential(
                    *[
                        nn.Conv2d(
                            in_ch,
                            out_ch,
                            kernel_size,
                            stride=1,
                            padding=kernel_size // 2),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                    ]
                )
            )
            in_ch = out_ch
        dec_blocks.append(
            nn.Sequential(
                *[
                    nn.Conv2d(
                        in_ch,
                        in_chans,
                        kernel_size,
                        stride=1,
                        padding=kernel_size // 2),
                ]
            )
        )
        self.decoder_blocks = nn.Sequential(*dec_blocks)

    def forward(self, x: torch.Tensor):
        y = self.encoder_blocks(x)
        z = self.decoder_blocks(y)
        return z

