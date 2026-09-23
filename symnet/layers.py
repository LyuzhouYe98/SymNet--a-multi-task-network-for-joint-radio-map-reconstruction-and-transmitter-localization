"""SymNet building blocks; attribute names preserve historical checkpoint keys."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last=True, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            return x * self.weight + self.bias
        return x * self.weight[:, None, None] + self.bias[:, None, None]


def convert_layer_norm_to_dynamic_tanh(module: nn.Module) -> nn.Module:
    output = module
    if isinstance(module, nn.LayerNorm):
        output = DynamicTanh(module.normalized_shape, channels_last=True)
    for name, child in module.named_children():
        output.add_module(name, convert_layer_norm_to_dynamic_tanh(child))
    return output


class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, hidden_channels):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, gate, skip):
        weights = self.psi(F.relu(self.W_gate(gate) + self.W_x(skip), inplace=True))
        return skip * weights


class SkipEncoder(nn.Module):
    def __init__(self, in_channels=3, bottleneck_channels=4, width=27, negative_slope=0.3):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, width, 3, padding="same")
        self.conv2d_1 = nn.Conv2d(width, width, 3, padding="same")
        self.conv2d_2 = nn.Conv2d(width, width, 3, padding="same")
        self.conv2d_3 = nn.Conv2d(width, width, 3, padding="same")
        self.conv2d_4 = nn.Conv2d(width, width, 3, padding="same")
        self.conv2d_5 = nn.Conv2d(width, width, 3, padding="same")
        self.conv2d_6 = nn.Conv2d(width, width, 3, padding="same")
        self.conv2d_7 = nn.Conv2d(width, width, 3, padding="same")
        self.conv2d_8 = nn.Conv2d(width, width, 3, padding="same")
        self.mu = nn.Conv2d(width, bottleneck_channels, 3, padding="same")
        self.average_pooling2d = nn.AvgPool2d(2)
        self.average_pooling2d_1 = nn.AvgPool2d(2)
        self.average_pooling2d_2 = nn.AvgPool2d(2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.leaky_relu(self.conv2d(x))
        x = self.leaky_relu(self.conv2d_1(x))
        x = self.leaky_relu(self.conv2d_2(x))
        skip1 = x
        x = self.average_pooling2d(x)
        x = self.leaky_relu(self.conv2d_3(x))
        x = self.leaky_relu(self.conv2d_4(x))
        x = self.leaky_relu(self.conv2d_5(x))
        skip2 = x
        x = self.average_pooling2d_1(x)
        x = self.leaky_relu(self.conv2d_6(x))
        x = self.leaky_relu(self.conv2d_7(x))
        x = self.leaky_relu(self.conv2d_8(x))
        skip3 = x
        x = self.average_pooling2d_2(x)
        return self.leaky_relu(self.mu(x)), skip1, skip2, skip3


class SkipDecoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, width=27, negative_slope=0.3):
        super().__init__()
        self.conv2d_transpose = nn.ConvTranspose2d(in_channels, in_channels, 3, padding=1)
        self.conv2d_transpose_1 = nn.ConvTranspose2d(in_channels + width, width, 3, padding=1)
        self.conv2d_transpose_2 = nn.ConvTranspose2d(width, width, 3, padding=1)
        self.conv2d_transpose_3 = nn.ConvTranspose2d(width, width, 3, padding=1)
        self.conv2d_transpose_4 = nn.ConvTranspose2d(2 * width, width, 3, padding=1)
        self.conv2d_transpose_5 = nn.ConvTranspose2d(width, width, 3, padding=1)
        self.conv2d_transpose_6 = nn.ConvTranspose2d(width, width, 3, padding=1)
        self.conv2d_transpose_7 = nn.ConvTranspose2d(2 * width, width, 3, padding=1)
        self.conv2d_transpose_8 = nn.ConvTranspose2d(width, width, 3, padding=1)
        self.conv2d_transpose_9 = nn.ConvTranspose2d(width, width, 3, padding=1)
        self.att = AttentionGate(in_channels, width, 128)
        self.att1 = AttentionGate(width, width, 64)
        self.att2 = AttentionGate(width, width, 32)
        self.up_sampling2d = nn.Upsample(scale_factor=2, mode="bilinear")
        self.up_sampling2d_1 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.up_sampling2d_2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv2d_output = nn.Conv2d(width, out_channels, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x, skip1, skip2, skip3):
        x = self.up_sampling2d(self.leaky_relu(self.conv2d_transpose(x)))
        x = torch.cat((x, self.att(x, skip3)), dim=1)
        x = self.leaky_relu(self.conv2d_transpose_1(x))
        x = self.leaky_relu(self.conv2d_transpose_2(x))
        x = self.up_sampling2d_1(self.leaky_relu(self.conv2d_transpose_3(x)))
        x = torch.cat((x, self.att1(x, skip2)), dim=1)
        x = self.leaky_relu(self.conv2d_transpose_4(x))
        x = self.leaky_relu(self.conv2d_transpose_5(x))
        x = self.up_sampling2d_2(self.leaky_relu(self.conv2d_transpose_6(x)))
        x = torch.cat((x, self.att2(x, skip1)), dim=1)
        x = self.leaky_relu(self.conv2d_transpose_7(x))
        x = self.leaky_relu(self.conv2d_transpose_8(x))
        x = self.leaky_relu(self.conv2d_transpose_9(x))
        return self.conv2d_output(x)


class SkipNet(nn.Module):
    def __init__(self, in_channels=3, bottleneck_channels=4, out_channels=1, width=27):
        super().__init__()
        self.encoder = SkipEncoder(in_channels, bottleneck_channels, width)
        self.decoder = SkipDecoder(bottleneck_channels, out_channels, width)

    def forward(self, x):
        x, skip1, skip2, skip3 = self.encoder(x)
        return self.decoder(x, skip1, skip2, skip3)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=16):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, guide):
        attended, _ = self.attn(query=x, key=guide, value=guide)
        return self.norm(x + attended)
