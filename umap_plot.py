from tqdm import tqdm

import umap
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ebm_modules import Backbone

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load model
# -----------------------
model = Backbone()
model_states = torch.load("weights_ssl/model-100.pt", map_location=device)
model.load_state_dict(model_states['encoder'])
model.to(device)
model.eval()

# -----------------------
# Load MNIST
# -----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
])

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

# -----------------------
# Extract embeddings
# -----------------------
embeddings = []
labels = []

embeddings = []
labels = []

with torch.no_grad():
    for img, target in tqdm(test_loader, desc="Extracting embeddings"):
        img = img.to(device)

        emb = model(img)

        embeddings.append(emb.cpu())
        labels.append(target)

embeddings = torch.cat(embeddings, dim=0)

# Normalize (important for UMAP + contrastive models)
embeddings = F.normalize(embeddings, dim=1)

embeddings = embeddings.numpy()
labels = torch.cat(labels, dim=0).numpy()

# -----------------------
# UMAP
# -----------------------
reducer = umap.UMAP(n_components=2, random_state=42)
emb_2d = reducer.fit_transform(embeddings)

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    emb_2d[:, 0],
    emb_2d[:, 1],
    # c=labels,
    cmap="tab10",
    s=5
)

plt.colorbar(scatter)
plt.title("UMAP of SSL Embeddings")
plt.show()