import os

import torch
import torch.nn as nn
from tqdm import tqdm


def train_ssl(
    encoder,
    target_model,
    train_loader,
    optimizer_func,
    loss_func,
    device,
    transform,
    epochs,
    save_path,
    save_interval,
    scheduler,
):
    """Train encoder using self-supervised learning with EMA target."""
    os.makedirs(save_path, exist_ok=True)

    ema = 0.99

    encoder.to(device)
    target_model.to(device)

    encoder.train()
    target_model.eval()

    # -- freeze target model
    for p in target_model.parameters():
        p.requires_grad = False

    optimizer = optimizer_func(encoder.parameters(), lr=1e-4)

    if scheduler:
        # -- cosine LR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )

    for epoch in range(epochs):
        process_bar = tqdm(train_loader, desc=f"SSL | Epoch {epoch+1}/{epochs}")
        epoch_loss = 0

        for img, _ in process_bar:
            # -- two augmented views
            img1 = transform(img)
            img2 = transform(img)

            img1 = img1.to(device)
            img2 = img2.to(device)

            x = encoder(img1)

            with torch.no_grad():
                y = target_model(img2)

            loss = loss_func(x, y)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # -- EMA update
            with torch.no_grad():
                for o, t in zip(encoder.parameters(), target_model.parameters()):
                    t.data.mul_(ema).add_(o.data, alpha=1 - ema)

            process_bar.set_postfix(
                {
                    "Loss": epoch_loss / len(train_loader),
                    "LR": optimizer.param_groups[0]["lr"],
                }
            )

        if scheduler:
            scheduler.step()

        if (epoch + 1) % save_interval == 0:
            # -- save target encoder
            torch.save(
                {
                    "encoder": target_model.state_dict(),
                },
                f"{save_path}/model-{epoch+1}.pt",
            )
            print(f"Model saved (epoch: {epoch + 1})")


def train_linear_probe(
    encoder,
    linear_probe,
    train_loader,
    optimizer_func,
    loss_func,
    device,
    epochs,
    save_path,
    save_interval,
    scheduler,
):
    """Train linear classifier on frozen encoder."""
    os.makedirs(save_path, exist_ok=True)

    encoder.to(device)
    linear_probe.to(device)

    optimizer = optimizer_func(linear_probe.parameters(), lr=1e-4)

    encoder.eval()
    # -- freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False

    linear_probe.train()

    if scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )

    for epoch in range(epochs):
        process_bar = tqdm(
            train_loader, desc=f"Linear Probe | Epoch {epoch+1}/{epochs}"
        )
        epoch_loss = 0

        for img, target in process_bar:
            img, target = img.to(device), target.to(device)

            with torch.no_grad():
                embed = encoder(img)

            out = linear_probe(embed)
            loss = loss_func(out, target)

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            process_bar.set_postfix(
                {
                    "Loss": epoch_loss / len(train_loader),
                    "LR": optimizer.param_groups[0]["lr"],
                }
            )

        if scheduler:
            scheduler.step()

        if (epoch + 1) % save_interval == 0:
            # -- save probe + encoder
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "linear_probe": linear_probe.state_dict(),
                },
                f"weights_ssl_linear_probe/model-{epoch+1}.pt",
            )
            print(f"Model saved (epoch: {epoch + 1})")


def test_linear_probe(encoder, linear_probe, test_loader, device):
    """Evaluate linear probe accuracy."""
    encoder.to(device)
    linear_probe.to(device)

    encoder.eval()
    linear_probe.eval()

    loss_func = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for img, target in tqdm(test_loader, desc="TESTING SSL"):
            img, target = img.to(device), target.to(device)

            embed = encoder(img)
            out = linear_probe(embed)

            loss = loss_func(out, target)
            total_loss += loss.item()

            preds = torch.argmax(out, dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

    print(f"Loss: {total_loss/len(test_loader):.4f}")
    print(f"Acc: {100*correct/total:.2f}%")


def train_supervised(
    encoder,
    classifier,
    train_loader,
    optimizer_func,
    loss_func,
    device,
    epochs,
    save_path,
    save_interval,
    scheduler=False,
):
    """Train encoder and classifier jointly (supervised)."""
    os.makedirs(save_path, exist_ok=True)

    encoder.to(device)
    classifier.to(device)

    encoder.train()
    classifier.train()

    optimizer = optimizer_func(
        list(encoder.parameters()) + list(classifier.parameters()), lr=1e-4
    )

    if scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )

    for epoch in range(epochs):
        process_bar = tqdm(train_loader, desc=f"Sup | Epoch {epoch+1}/{epochs}")
        epoch_loss = 0

        for img, target in process_bar:
            img, target = img.to(device), target.to(device)

            embed = encoder(img)
            out = classifier(embed)

            loss = loss_func(out, target)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            process_bar.set_postfix(
                {
                    "Loss": epoch_loss / len(train_loader),
                    "LR": optimizer.param_groups[0]["lr"],
                }
            )

        if scheduler:
            scheduler.step()

        if (epoch + 1) % save_interval == 0:
            # -- save model
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                },
                f"{save_path}/model-{epoch+1}.pt",
            )
            print(f"Model saved (epoch: {epoch + 1})")


def test_supervised(encoder, classifier, test_loader, device):
    """Evaluate supervised model."""
    encoder.to(device)
    classifier.to(device)

    encoder.eval()
    classifier.eval()

    loss_func = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for img, target in tqdm(test_loader, desc="Testing supervised learning"):
            img, target = img.to(device), target.to(device)

            embed = encoder(img)
            out = classifier(embed)

            loss = loss_func(out, target)
            total_loss += loss.item()

            preds = torch.argmax(out, dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)

    print(f"Loss: {total_loss/len(test_loader):.4f}")
    print(f"Acc: {100*correct/total:2.2f}%")
