import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation, PillowWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from umap import UMAP  # UMAP

from modules import Backbone
from vit import ViT  # SSL ViT model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(weight_path, model_type="ssl"):
    if model_type == "ssl_vit":
        model = ViT(
            image_size=28,
            patch_size=7,
            num_classes=-1,
            dim=64,
            depth=3,
            heads=4,
            mlp_dim=128,
            pool="mean",
            channels=1,
            dim_head=64,
            dropout=0.0,
            emb_dropout=0.0,
        ).to(device)
    elif model_type in ["sup", "ssl"]:
        model = Backbone().to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state["encoder"])
    model.eval()
    return model


def extract_embeddings(model, loader):
    embeddings = []
    labels = []
    with torch.no_grad():
        for img, target in tqdm(loader, desc="Extracting embeddings"):
            img = img.to(device)
            emb = model(img)
            embeddings.append(emb.cpu())
            labels.append(target)
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    embeddings = F.normalize(embeddings, dim=1).numpy()
    labels = labels.numpy()
    return embeddings, labels


def main(animation=False, model_type="ssl", weight=None, duration=5):
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    # -- single model mode
    if not animation:
        if weight is None:
            raise ValueError("Weight must be provided")
        weight_path = f"{model_type}/weights/{weight}"
        model = load_model(weight_path, model_type)
        embeddings, labels = extract_embeddings(model, test_loader)

        # -- UMAP
        umap_proj = UMAP(n_components=2, random_state=42)
        emb_2d = umap_proj.fit_transform(embeddings)

        # -- plot
        fig, ax = plt.subplots(figsize=(9, 7))
        cmap = plt.get_cmap("tab10")
        unique_labels = np.unique(labels)

        for i, lab in enumerate(unique_labels):
            idx = labels == lab
            ax.scatter(
                emb_2d[idx, 0],
                emb_2d[idx, 1],
                s=4,
                color=cmap(i),
                label=str(lab),
                alpha=0.9,
                edgecolors="none",
            )

        ax.set_title(f"UMAP Embeddings ({weight})", fontsize=16, pad=15)
        ax.set_xlabel("Dim 1", fontsize=11, labelpad=10)
        ax.set_ylabel("Dim 2", fontsize=11, labelpad=10)
        ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.15)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        ax.legend(
            title="Digit",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=9,
            title_fontsize=10,
            markerscale=2,
        )
        plt.subplots_adjust(right=0.8)
        plt.savefig(os.path.join(model_type, f"umap_{weight}.png"), dpi=300)
        plt.close()

    # -- animation mode
    else:
        folder_path = f"{model_type}/weights"
        weight_files = [f for f in os.listdir(folder_path) if f.endswith(".pt")]

        def extract_epoch(filename):
            return int(filename.split("-")[1].split(".")[0])

        weight_files = sorted(weight_files, key=extract_epoch)

        all_emb_2d = []
        all_labels = None

        print("Precomputing UMAP for each model...")
        for w in weight_files:
            print(f"Processing {w}")
            weight_path = os.path.join(folder_path, w)
            model = load_model(weight_path, model_type)
            emb, labels = extract_embeddings(model, test_loader)

            umap_proj = UMAP(n_components=2, random_state=42)
            emb_2d = umap_proj.fit_transform(emb)
            all_emb_2d.append(emb_2d)

            if all_labels is None:
                all_labels = labels

        # -- animation plot
        fig, ax = plt.subplots(figsize=(9, 7))
        cmap = plt.get_cmap("tab10")
        unique_labels = np.unique(all_labels)

        scatters = []
        for i, lab in enumerate(unique_labels):
            idx = all_labels == lab
            sc = ax.scatter(
                all_emb_2d[0][idx, 0],
                all_emb_2d[0][idx, 1],
                s=4,
                color=cmap(i),
                label=str(lab),
                alpha=0.9,
                edgecolors="none",
            )
            scatters.append((sc, idx))

        title_text = ax.set_title(f"UMAP Embeddings ({weight_files[0]})", fontsize=14)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.15)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        ax.legend(
            title="Digit",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=9,
            title_fontsize=10,
            markerscale=2,
        )
        plt.subplots_adjust(right=0.8)

        # -- global axes
        all_x = np.concatenate([emb[:, 0] for emb in all_emb_2d])
        all_y = np.concatenate([emb[:, 1] for emb in all_emb_2d])
        x_pad = 0.05 * (all_x.max() - all_x.min())
        y_pad = 0.05 * (all_y.max() - all_y.min())
        ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
        ax.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)

        def update(frame):
            emb = all_emb_2d[frame]
            for sc, idx in scatters:
                sc.set_offsets(emb[idx])
            title_text.set_text(f"UMAP Embeddings ({weight_files[frame]})")
            return [s[0] for s in scatters] + [title_text]

        num_frames = len(all_emb_2d)
        interval = (duration * 1000) / num_frames
        fps = num_frames / duration

        anim = FuncAnimation(
            fig, update, frames=len(all_emb_2d), interval=interval, blit=True
        )
        anim_path = os.path.join(model_type, "umap_animation.gif")
        anim.save(anim_path, writer=PillowWriter(fps=fps))
        print(f"Animation saved to {anim_path}")


if __name__ == "__main__":
    # main(animation=False, model_type="ssl_vit", weight="model-300.pt")
    main(animation=True, model_type="ssl_vit", duration=5)

    # main(animation=False, model_type="ssl", weight="model-300.pt")
    main(animation=True, model_type="ssl", duration=5)

    # main(animation=False, model_type="sup", weight="model-50.pt")
    main(animation=True, model_type="sup", duration=5)
