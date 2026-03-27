import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from ebm_modules import Backbone

# -- device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- load pretrained encoder
model_mode = "weights_ssl"
num_epoch = "300"

# model_mode = "weights_supervised"
# num_epoch = "50"

model = Backbone()
model_states = torch.load(
    model_mode + "/model-" + num_epoch + ".pt", map_location=device
)
model.load_state_dict(model_states["encoder"])
model.to(device)
model.eval()

# -- MNIST dataset
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

test_data = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

# -- extract embeddings
embeddings = []
labels = []

with torch.no_grad():
    for img, target in tqdm(test_loader, desc="Extracting embeddings"):
        img = img.to(device)

        emb = model(img)

        embeddings.append(emb.cpu())
        labels.append(target)

# -- concatenate batches
embeddings = torch.cat(embeddings, dim=0)
labels = torch.cat(labels, dim=0)

# -- normalize embeddings
embeddings = F.normalize(embeddings, dim=1)

# -- convert to numpy
embeddings = embeddings.numpy()
labels = labels.numpy()

# -- t-SNE projection
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    random_state=42,
    init="pca",
)

emb_2d = tsne.fit_transform(embeddings)

# -- plot
plt.figure(figsize=(8, 6))

scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="tab10", s=5)

plt.colorbar(scatter)
# plt.title("t-SNE of SSL Embeddings (MNIST)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.savefig(f"tsne_{model_mode}_e{num_epoch}.png")
plt.show()
