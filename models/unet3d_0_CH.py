import torch
from torch import nn


class Conv3dCH(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
        super(Conv3dCH, self).__init__()
        self.convx = nn.Conv3d(in_channels, out_channels, (kernel_size[0], 1, kernel_size[2]), stride=stride, padding=(padding, 0, padding), dilation=dilation, groups=groups, bias=False)
        self.convy = nn.Conv3d(in_channels, out_channels, (kernel_size[0], kernel_size[1], 1), stride=stride, padding=(padding, padding, 0), dilation=dilation, groups=groups, bias=False)
        self.convz = nn.Conv3d(in_channels, out_channels, (1, kernel_size[1], kernel_size[2]), stride=stride, padding=(0, padding, padding), dilation=dilation, groups=groups, bias=False)

    def forward(self, x):
        cx = self.convx(x)
        cy = self.convy(x)
        cz = self.convz(x)
        out = cx
        out.add_(cy)
        out.add_(cz)
        return out


class ConvTranspose3dCH(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=1, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None):
        super(ConvTranspose3dCH, self).__init__()
        self.convx = nn.ConvTranspose3d(in_channels, out_channels, (kernel_size[0], 1, kernel_size[2]), stride=stride, padding=(padding, 0, padding), dilation=dilation, output_padding=1, groups=groups, bias=False)
        self.convy = nn.ConvTranspose3d(in_channels, out_channels, (kernel_size[0], kernel_size[1], 1), stride=stride, padding=(padding, padding, 0), dilation=dilation, output_padding=1, groups=groups, bias=False)
        self.convz = nn.ConvTranspose3d(in_channels, out_channels, (1, kernel_size[1], kernel_size[2]), stride=stride, padding=(0, padding, padding), dilation=dilation, output_padding=1, groups=groups, bias=False)

    def forward(self, x):
        cx = self.convx(x)
        cy = self.convy(x)
        cz = self.convz(x)
        out = cx
        out.add_(cy)
        out.add_(cz)
        return out


class UNet(nn.Module):
    '''
    starting general setup:
    :param in_channels = 1: number of input channels
    :pram init_feat = 32: desired number of output channels
    :param kernel_size = 3:
    '''

    def _block(self, in_channels, mid_channels, out_channels, kernel_size=(3, 3, 3), dtype=torch.float32):
        return torch.nn.Sequential(
            Conv3dCH(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
                dtype=dtype,
            ),
            torch.nn.BatchNorm3d(
                num_features=mid_channels
            ),
            torch.nn.ReLU(
                inplace=True
            ),
            Conv3dCH(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
                dtype=dtype,
            ),
            torch.nn.BatchNorm3d(num_features=out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def final_block(self, in_channels, mid_channels, out_channels, kernel_size=(3, 3, 3), dtype=torch.float32):
        return torch.nn.Sequential(
            Conv3dCH(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
                dtype=dtype,
            ),
            torch.nn.BatchNorm3d(num_features=mid_channels),
            torch.nn.ReLU(inplace=True),
            Conv3dCH(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
                dtype=dtype,
            ),
            torch.nn.BatchNorm3d(num_features=mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(  # in the last step fot a (1,1,1) is not necessary to use a CH filter
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=(1, 1, 1),
                padding=0,
                bias=False,
                dtype=dtype,
            ),
            torch.nn.Softmax(dim=1),  # Sigmoid is used for binary classification methods where we only have 2 classes, while SoftMax applies to multiclass problems
        )

    def __init__(self, n_classes: int = 1):
        super(UNet, self).__init__()
        init_feat = 32
        in_channels = 1
        out_channels = n_classes

        # Encode
        self.conv_encoder1 = self._block(in_channels=in_channels, mid_channels=init_feat, out_channels=init_feat * 2)
        self.conv_maxpool1 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv_encoder2 = self._block(init_feat * 2, init_feat * 2, init_feat * 4)
        self.conv_maxpool2 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.conv_encoder3 = self._block(init_feat * 4, init_feat * 4, init_feat * 8)
        self.conv_maxpool3 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))

        # Bottleneck
        self.bottleneck = self._block(init_feat * 8, init_feat * 8, init_feat * 16)

        # Decode
        self.upconv3 = ConvTranspose3dCH(in_channels=init_feat * 16, out_channels=init_feat * 16, kernel_size=(3, 3, 3), stride=2, padding=1, output_padding=1, bias=False)
        self.conv_decoder3 = self._block(init_feat * 16 + init_feat * 8, init_feat * 8, init_feat * 8)
        self.upconv2 = ConvTranspose3dCH(in_channels=init_feat * 8, out_channels=init_feat * 8, kernel_size=(3, 3, 3), stride=2, padding=1, output_padding=1, bias=False)
        self.conv_decoder2 = self._block(init_feat * 8 + init_feat * 4, init_feat * 4, init_feat * 4)
        self.upconv1 = ConvTranspose3dCH(in_channels=init_feat * 4, out_channels=init_feat * 4, kernel_size=(3, 3, 3), stride=2, padding=1, output_padding=1, bias=False)
        self.conv_decoder1 = self.final_block(init_feat * 4 + init_feat * 2, init_feat * 2, out_channels=out_channels)

    def forward(self, x):
        '''
        :param x: input tensor to be convolved
        '''

        # Encode:
        enc1 = self.conv_encoder1(x)
        enc2 = self.conv_encoder2(self.conv_maxpool1(enc1))
        enc3 = self.conv_encoder3(self.conv_maxpool2(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.conv_maxpool3(enc3))

        # Decode
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.conv_decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.conv_decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.conv_decoder1(dec1)
        return dec1
