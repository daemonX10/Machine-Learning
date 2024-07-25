"""
Discriminator and Generator implementation from DCGAN paper

Usage:
* 2020-11-01: Initial coding
* 2022-12-20: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Discriminator network from the DCGAN paper. Takes an image and outputs a scalar value indicating
    whether the input image is real or fake.
    """
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # Input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # Adding convolutional blocks
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block layers, the image output is 4x4
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),  # Output is a single scalar value
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Helper function to create a convolutional block with Conv2D, BatchNorm, and LeakyReLU layers.
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            # Uncomment the following line if BatchNorm is desired
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    """
    Generator network from the DCGAN paper. Takes a noise vector and outputs an image.
    """
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # Output: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # Output: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # Output: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # Output: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),  # Output: 64x64
            nn.Tanh(),  # Output activation function
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Helper function to create a deconvolutional block with ConvTranspose2D, BatchNorm, and ReLU layers.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            # Uncomment the following line if BatchNorm is desired
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    """
    Initializes weights of the model using normal distribution as per the DCGAN paper.
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    """
    Tests the Discriminator and Generator classes to ensure they produce the expected output shapes.
    """
    N, in_channels, H, W = 8, 3, 64, 64  # Batch size, image channels, height, width
    noise_dim = 100  # Dimension of the noise vector
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, tests passed!")


if __name__ == "__main__":
    test()
