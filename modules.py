import torch
import torch.nn as nn
import torch.nn.functional as F

"""Loss function can be obtained from torch.nn or created as an separated function.
I integrated with the backbone for my own convenience."""


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
            nn.Linear(36, 48),
            act_layer(),
            nn.Linear(48, 64),
            act_layer(),
            nn.Linear(64, 64),
        )

    def forward(self, x):
        """Compute embedding from input."""
        out = self.conv_layers(x)
        out = self.ffn_layers(out)
        return out


class ClassificationHead(nn.Module):
    def __init__(self, act_layer=nn.ReLU, input_dim=64):
        super().__init__()

        self.hidden_layers = 2
        self.neurons = 64

        layers = []
        layers.append(nn.Linear(input_dim, self.neurons))
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
class LinearProbe(nn.Module):
    def __init__(self, input_dim=64):
        super().__init__()

        # -- project to 10 classes
        self.probe = nn.Linear(input_dim, 10)

    def forward(self, x):
        """Linear classification on frozen features."""
        return self.probe(x)


class SSLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

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
