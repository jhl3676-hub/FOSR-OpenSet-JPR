import torch
import torch.nn as nn


# ==================== Complex Autoencoder ====================

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)
        self.fc_i = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Assumes input is [Batch, 2*Length], splits into Real and Imag parts
        a = int(x.shape[1] / 2)
        x_r, x_i = torch.split(x, a, dim=1)
        out_r = self.fc_r(x_r) - self.fc_i(x_i)
        out_i = self.fc_r(x_i) + self.fc_i(x_r)
        return torch.cat((out_r, out_i), dim=-1)


class ComplexAutoencoder(nn.Module):
    def __init__(self, input_dim=128, encoding_dim=128):
        super(ComplexAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            ComplexLinear(input_dim, encoding_dim),
            nn.ReLU(),
            ComplexLinear(encoding_dim, encoding_dim // 2),
        )
        self.decoder = nn.Sequential(
            ComplexLinear(encoding_dim // 2, encoding_dim),
            nn.ReLU(),
            ComplexLinear(encoding_dim, input_dim),
        )

    def forward(self, x):
        if x.dim() == 3:
            B, C, L = x.shape
            x = x.reshape(B, -1)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.reshape(-1, 2, 128)
        return encoded, decoded


# ==================== FOSR Components ====================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.same_shape = in_channels == out_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if not self.same_shape:
            self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            identity = self.res_conv(identity)
        return self.relu(out + identity)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.block1 = ResidualBlock(3, 32)
        self.block2 = ResidualBlock(32, 64)
        self.block3 = ResidualBlock(64, 128)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.expand = nn.Linear(128, 128)
        self.block1 = ResidualBlock(128, 64)
        self.block2 = ResidualBlock(64, 32)
        self.block3 = ResidualBlock(32, 3)

    def forward(self, x):
        x = self.expand(x).unsqueeze(-1).repeat(1, 1, 128)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class Critic(nn.Module):
    def __init__(self, feat_dim=128, hidden_dims=[256, 128, 64]):
        super(Critic, self).__init__()
        layers = []
        in_dim = feat_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class FOSR(nn.Module):
    def __init__(self, feat_dim=128):
        super(FOSR, self).__init__()
        self.encoder = Encoder()
        self.recon_decoder = Decoder()
        self.critic = Critic(feat_dim)

    def forward(self, x):
        features = self.encoder(x)
        recon = self.recon_decoder(features)
        critic_scores = self.critic(features)
        return features, recon, critic_scores