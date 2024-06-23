import torch
import torch.nn as nn
import torch.nn.functional as F


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"


class ConvDeconv(nn.Module):
    def __init__(self):
        super().__init__()  # x 256 256 256 1
        self.conv3d_s2_1 = nn.Conv3d(1, 64, 4, 2)  # x128
        self.conv3d_s1_1 = nn.Conv3d(64, 64, 1, 1)

        self.conv3d_s2_2 = nn.Conv3d(64, 128, 4, 2)  # x64
        self.conv3d_s1_2 = nn.Conv3d(128, 128, 1, 1)

        self.conv3d_s2_3 = nn.Conv3d(128, 512, 4, 2)  # x32
        self.conv3d_s1_3 = nn.Conv3d(512, 512, 1, 1)

        self.conv3d_s2_4 = nn.Conv3d(512, 1024, 4, 2)  # x16    # -1 1024 16 16 16
        self.conv3d_s1_4 = nn.Conv3d(1024, 1024, 1, 1)

        # self.reshape = torch.Tensor.reshape((-1, 1024, 64, 64))

        self.deconv2d_s2_1 = nn.ConvTranspose2d(1024, 512, 4, 2, padding=1)  # x128
        self.deconv2d_s1_1 = nn.ConvTranspose2d(512, 512, 1, 1)

        self.deconv2d_s2_2 = nn.ConvTranspose2d(512, 1, 4, 2, padding=1)  # 256

    def forward(self, x):
        # CONV:
        x = F.pad(x, (1, 1, 1, 1, 1, 1), mode="constant")
        x = self.conv3d_s2_1(x)
        x = nn.LeakyReLU()(x)
        x = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)(x)

        x = self.conv3d_s1_1(x)
        x = nn.LeakyReLU()(x)

        x = F.pad(x, (1, 1, 1, 1, 1, 1), mode="constant")
        x = self.conv3d_s2_2(x)
        x = nn.LeakyReLU()(x)
        x = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)(x)

        x = self.conv3d_s1_2(x)
        x = nn.LeakyReLU()(x)

        x = F.pad(x, (1, 1, 1, 1, 1, 1), mode="constant")
        x = self.conv3d_s2_3(x)
        x = nn.LeakyReLU()(x)
        x = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)(x)

        x = self.conv3d_s1_3(x)
        x = nn.LeakyReLU()(x)

        x = F.pad(x, (1, 1, 1, 1, 1, 1), mode="constant")
        x = self.conv3d_s2_4(x)
        x = nn.LeakyReLU()(x)
        x = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)(x)

        x = self.conv3d_s1_4(x)
        x = nn.LeakyReLU()(x)

        x = torch.reshape(x, [-1, 1024, 64, 64])

        # DECONV:
        x = self.deconv2d_s2_1(x)
        x = nn.LeakyReLU()(x)

        x = self.deconv2d_s1_1(x)
        x = nn.LeakyReLU()(x)

        x = self.deconv2d_s2_2(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary
    model = ConvDeconv()
    summary(model, input_size=(5, 1, 256, 256, 256))
