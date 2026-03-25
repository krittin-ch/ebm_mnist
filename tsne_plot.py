from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from ebm_modules import Backbone


# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load model
# -----------------------
model = Backbone()

model_states = torch.load("weights_ssl/model-250.pt", map_location=device)
model.load_state_dict(model_states["encoder"])

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

with torch.no_grad():
    for img, target in tqdm(test_loader, desc="Extracting embeddings"):
        img = img.to(device)

        emb = model(img)

        embeddings.append(emb.cpu())
        labels.append(target)

# Concatenate
embeddings = torch.cat(embeddings, dim=0)
labels = torch.cat(labels, dim=0)

# Normalize (important)
embeddings = F.normalize(embeddings, dim=1)

# Convert to numpy
embeddings = embeddings.numpy()
labels = labels.numpy()

# -----------------------
# t-SNE
# -----------------------
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    # n_iter=1000,
    random_state=42,
    init="pca"
)

emb_2d = tsne.fit_transform(embeddings)

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    emb_2d[:, 0],
    emb_2d[:, 1],
    c=labels,
    cmap="tab10",
    s=5
)

plt.colorbar(scatter)
plt.title("t-SNE of SSL Embeddings (MNIST)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")

plt.show()