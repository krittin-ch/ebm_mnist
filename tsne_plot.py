import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from modules import Backbone
from vit import ViT  # SSL ViT model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Function to load a model based on type
# -----------------------------
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
    elif model_type == "sup" or model_type == "ssl":
        model = Backbone().to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state["encoder"])
    model.eval()
    return model


# -----------------------------
# Extract embeddings
# -----------------------------
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


# -----------------------------
# Main function
# -----------------------------
def main(animation=False, model_type="ssl", weight=None, interval=1000):
    # MNIST test set
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    if not animation:
        # Single model mode
        if weight is None:
            raise ValueError("Weight must be provided for single model mode")
        weight_path = f"{model_type}/weights/{weight}"
        model = load_model(weight_path, model_type)
        embeddings, labels = extract_embeddings(model, test_loader)

        # t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate=200,
            random_state=42,
            init="pca",
        )
        emb_2d = tsne.fit_transform(embeddings)

        # Plot
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="tab10", s=5)
        plt.colorbar(scatter)
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.title(f"t-SNE Embeddings ({weight})")
        plt.savefig(os.path.join(model_type, f"tsne_{weight}.png"))
        # plt.show()

    else:
        # Animation mode: iterate all .pt files in folder
        folder_path = f"{model_type}/weights"
        weight_files = [f for f in os.listdir(folder_path) if f.endswith(".pt")]

        def extract_epoch(filename):
            return int(filename.split("-")[1].split(".")[0])

        weight_files = sorted(weight_files, key=extract_epoch)

        all_emb_2d = []
        all_labels = None

        print("Precomputing t-SNE for each model...")
        for w in weight_files:
            print(f"Processing {w}")
            weight_path = os.path.join(folder_path, w)
            model = load_model(weight_path, model_type)
            emb, labels = extract_embeddings(model, test_loader)

            # Compute t-SNE independently per model
            tsne = TSNE(
                n_components=2,
                perplexity=30,
                learning_rate=200,
                random_state=42,
                init="pca",
            )
            emb_2d = tsne.fit_transform(emb)
            all_emb_2d.append(emb_2d)

            if all_labels is None:
                all_labels = labels

        # Animation
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            all_emb_2d[0][:, 0], all_emb_2d[0][:, 1], c=all_labels, cmap="tab10", s=5
        )
        title_text = ax.set_title(f"t-SNE Embeddings ({weight_files[0]})")
        plt.colorbar(scatter)
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")

        all_x = np.concatenate([emb[:, 0] for emb in all_emb_2d])
        all_y = np.concatenate([emb[:, 1] for emb in all_emb_2d])

        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()

        x_pad = 0.05 * (x_max - x_min)
        y_pad = 0.05 * (y_max - y_min)

        # Set static axes before animation
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        def update(frame):
            scatter.set_offsets(all_emb_2d[frame])
            title_text.set_text(f"t-SNE Embeddings ({weight_files[frame]})")
            return scatter, title_text

        anim = FuncAnimation(
            fig, update, frames=len(all_emb_2d), interval=interval, blit=True
        )
        anim_path = os.path.join(model_type, "tsne_animation.gif")
        anim.save(anim_path, writer=PillowWriter(fps=1))
        print(f"Animation saved to {anim_path}")
        # plt.show()


# -----------------------------
# Run main
# -----------------------------
if __name__ == "__main__":
    # main(animation=False, model_type="ssl_vit", weight="model-300.pt")
    # main(animation=True, model_type="ssl_vit", interval=100)

    main(animation=False, model_type="ssl", weight="model-300.pt")
    main(animation=True, model_type="ssl", interval=100)

    main(animation=False, model_type="sup", weight="model-50.pt")
    main(animation=True, model_type="sup")
