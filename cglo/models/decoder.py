import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(self, c_in:int, c_out:int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(c_out)
        )

    def forward(self, x):
        return self.block(x)

class ConvUp(nn.Module):

    def __init__(self, c_in:int, c_out:int, scale_factor=2, mode='nearest'):
        super().__init__()

        self.block = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode=mode),
            ConvBlock(c_in, c_out)
        )

    def forward(self, x):
        return self.block(x)

class LatentLinear(nn.Module):

    def __init__(self, c_in:int, c_out:int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(c_in, c_out, bias=False),
            nn.ReLU(True),
            nn.BatchNorm1d(c_out)
        )

    def forward(self, x):
        return self.block(x)

class Decoder(nn.Module):

    def __init__(self, layers, out_size):

        super().__init__()

        self.c_in = (4  * layers[1])
        self.c_out = layers[-1]

        self.mlp = LatentLinear(layers[0], self.c_in)

        net = nn.Sequential(
            *[ConvUp(i, o) for i, o in zip(layers[1:-2], layers[2:-1])]
        )

        scale_factor = out_size / 2**(len(layers)-2)
        out = nn.Sequential(
            ConvUp(layers[-2], layers[-1], scale_factor=scale_factor),
            nn.Conv2d(self.c_out, 1, 3, 1, 1)
        )

        sig = nn.Sigmoid()

        self.dec = nn.Sequential(net, out, sig)

    def forward(self, x):
        return self.dec(self.mlp(x).view(x.shape[0], self.c_in//4, 2, 2))

