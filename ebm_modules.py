import torch
import torch.nn as nn
import torch.nn.functional as F


class Backbone(nn.Module):
    def __init__(self,act_layer=nn.ReLU):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1),
            act_layer(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1),
            act_layer(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1),
            nn.Flatten()
        )

        self.ffn_layers = nn.Sequential(
            nn.Linear(36, 64),
            act_layer(),
            nn.Linear(64, 128),
            act_layer(),
            nn.Linear(128, 128),
        )

    def forward(self, x):
        out = self.conv_layers(x)
        out = self.ffn_layers(out)

        return out

    def energy(self, x_emb, y_emb):
        return torch.norm(x_emb - y_emb, p=2, dim=-1)
    
    def energy_loss(self, x_emb, y_emb):
        return self.energy(x_emb, y_emb).mean()

    def nll_loss(self, x_emb, y_emb, ybar_emb):
        Epos = self.energy(x_emb, y_emb)

        yall_emb = torch.cat([y_emb, ybar_emb], dim=0)
        Eall = self.energy(x_emb.unsqueeze(1), yall_emb.unsqueeze(0))

        beta = 0.25
        F = torch.logsumexp(-beta * Eall, dim=1) / beta # orch.log(torch.exp(-beta * Eall).sum(dim=1)) / beta # sum over y

        return (Epos + F).mean()
    
    def sq_sq_loss(self, x_emb, y_emb, ybar_emb):
        m = 1.0
        Epos = self.energy(x_emb, y_emb)
        Eneg = self.energy(x_emb, ybar_emb)
        loss = Epos.pow(2) + F.relu(m - Eneg).pow(2)
    
        return loss.mean()
    
    def contrastive_loss(self, x_emb, y_emb, temperature=0.05):
        # normalize embeddings (VERY IMPORTANT)
        x_emb = F.normalize(x_emb, dim=-1)
        y_emb = F.normalize(y_emb, dim=-1)

        # similarity matrix (B x B)
        logits = x_emb @ y_emb.T / temperature

        labels = torch.arange(x_emb.size(0), device=x_emb.device)

        loss_x_to_y = F.cross_entropy(logits, labels)
        loss_y_to_x = F.cross_entropy(logits.T, labels)

        return (loss_x_to_y + loss_y_to_x) / 2


class ClassificationHead(nn.Module):
    def __init__(self, act_layer=nn.ReLU):
        super().__init__()

        self.hidden_layers = 2
        self.neurons = 128

        layers = []
        layers.append(nn.Linear(128, self.neurons))
        layers.append(act_layer())

        for _ in range(self.hidden_layers):
            layers.append(nn.Linear(self.neurons, self.neurons))
            layers.append(act_layer())

        layers.append(nn.Linear(self.neurons, 10))
    
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
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

        for _ in range(self.hidden_layers):
            layers.append(nn.Linear(self.neurons, self.neurons))
            layers.append(act_layer())

        layers.append(nn.Linear(self.neurons, 128))
    
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()

        self.probe = nn.Linear(128, 10)

    def forward(self, x):
        return self.probe(x)
