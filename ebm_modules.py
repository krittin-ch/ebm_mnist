import torch
import torch.nn as nn
import torch.nn.functional as F


class Backbone(nn.Module):
    def __init__(self, act_layer=nn.ReLU):
        super().__init__()

        # -- conv feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1),
            act_layer(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1),
            act_layer(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1),
            nn.Flatten(),
        )

        # -- embedding MLP
        self.ffn_layers = nn.Sequential(
            nn.Linear(36, 64),
            act_layer(),
            nn.Linear(64, 128),
            act_layer(),
            nn.Linear(128, 128),
        )

    def forward(self, x):
        """Compute embedding from input."""
        out = self.conv_layers(x)
        out = self.ffn_layers(out)
        return out

    def energy(self, x_emb, y_emb):
        """L2 distance between embeddings."""
        return torch.norm(x_emb - y_emb, p=2, dim=-1)

    def energy_loss(self, x_emb, y_emb):
        """Mean pairwise energy."""
        return self.energy(x_emb, y_emb).mean()

    def nll_loss(self, x_emb, y_emb, ybar_emb):
        """Log-sum-exp based contrastive loss."""
        Epos = self.energy(x_emb, y_emb)
        yall_emb = torch.cat([y_emb, ybar_emb], dim=0)
        Eall = self.energy(x_emb.unsqueeze(1), yall_emb.unsqueeze(0))

        beta = 0.25
        F = torch.logsumexp(-beta * Eall, dim=1) / beta

        return (Epos + F).mean()

    def sq_sq_loss(self, x_emb, y_emb, ybar_emb):
        """Squared loss with margin on negatives."""
        m = 1.0
        Epos = self.energy(x_emb, y_emb)
        Eneg = self.energy(x_emb, ybar_emb)
        loss = Epos.pow(2) + F.relu(m - Eneg).pow(2)

        return loss.mean()

    def contrastive_loss(self, x_emb, y_emb, temperature=0.05):
        """Symmetric contrastive loss."""
        x_emb = F.normalize(x_emb, dim=-1)
        y_emb = F.normalize(y_emb, dim=-1)

        logits = x_emb @ y_emb.T / temperature
        labels = torch.arange(x_emb.size(0), device=x_emb.device)

        loss_x_to_y = F.cross_entropy(logits, labels)
        loss_y_to_x = F.cross_entropy(logits.T, labels)

        return (loss_x_to_y + loss_y_to_x) / 2

    def symmetric_decorrelation_loss(
        self, x_emb, y_emb, lambda_offdiag=0.005, eps=1e-4
    ):
        """Symmetric decorrelation loss for two views."""
        # -- normalize embeddings (zero-mean, unit-variance)
        x_norm = (x_emb - x_emb.mean(0)) / (x_emb.std(0) + eps)
        y_norm = (y_emb - y_emb.mean(0)) / (y_emb.std(0) + eps)

        # -- cross-correlation matrix
        c = x_norm.T @ y_norm / x_norm.size(0)

        # -- on-diagonal: alignment to 1
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

        # -- off-diagonal: decorrelation to 0
        def off_diagonal(x):
            n = x.size(0)
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

        off_diag = off_diagonal(c).pow(2).sum()

        # -- total loss
        loss = on_diag + lambda_offdiag * off_diag

        return loss


class ClassificationHead(nn.Module):
    def __init__(self, act_layer=nn.ReLU):
        super().__init__()

        self.hidden_layers = 2
        self.neurons = 128

        layers = []
        layers.append(nn.Linear(128, self.neurons))
        layers.append(act_layer())

        # -- hidden layers
        for _ in range(self.hidden_layers):
            layers.append(nn.Linear(self.neurons, self.neurons))
            layers.append(act_layer())

        layers.append(nn.Linear(self.neurons, 10))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Return class logits."""
        return self.layers(x)


# For SSL
class ProjectionHead(nn.Module):
    def __init__(self, act_layer=nn.ReLU):
        super().__init__()

        self.hidden_layers = 2
        self.neurons = 128

        layers = []
        layers.append(nn.Linear(128, self.neurons))
        layers.append(act_layer())

        # -- hidden layers
        for _ in range(self.hidden_layers):
            layers.append(nn.Linear(self.neurons, self.neurons))
            layers.append(act_layer())

        layers.append(nn.Linear(self.neurons, 128))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Project embeddings for SSL."""
        return self.layers(x)


class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()

        # project to 10 classes
        self.probe = nn.Linear(128, 10)

    def forward(self, x):
        """Linear classification on frozen features."""
        return self.probe(x)
