import torch
import ollama
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# Generate some example data

# print(X_pca)
model = "llama3.2"

# emd = ollama.embeddings(
#     model=model, 
#     prompt="Howdy how is life"
# )

# read json file
def read_date_from_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

cnn = read_date_from_json("CNN.json")
cnn = cnn["CNN"]
cnn_content = [x["content"] for x in cnn]
fox_news = read_date_from_json("foxnews.json")
fox_news = fox_news["foxnews"]
fox_news_content = [x["content"] for x in fox_news]

print("Starting to make embeddings")
# make embeddings then do pca on them
cnn_embeddings = torch.tensor([ollama.embeddings(model, x)["embedding"] for x in cnn_content])
fox_news_embeddings = torch.tensor([ollama.embeddings(model, x)["embedding"] for x in fox_news_content])

print("Done making embeddings")

# make embeddings one list
embeddings = torch.cat([cnn_embeddings, fox_news_embeddings], dim=0)
labels = ['Fox News'] * fox_news_embeddings.shape[0] + ['CNN'] * cnn_embeddings.shape[0]

# assert len(embeddings) == len(cnn_embeddings) + len(fox_news_embeddings)

# do pca on embeddings
torch_embeddings = torch.tensor(embeddings)

print("Starting PCA")
pca = PCA()
pca.fit(torch_embeddings)

# Transform the data to 2 dimensions for visualization by selecting the top 2 components
# reduced_embeddings = pca.transform(torch_embeddings)[:, :2]

# # Plot the reduced embeddings with colors for each source
# plt.figure(figsize=(10, 6))
# for source, color in zip(['Fox News', 'CNN'], ['red', 'blue']):
#     indices = [i for i, label in enumerate(labels) if label == source]
#     plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=source, alpha=0.6, color=color)

# plt.title("2D PCA of High-Dimensional Article Embeddings")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.legend()
# plt.savefig("pca_plot_2d.png", format="png", dpi=500)
# plt.show()
explained_variance_ratio = pca.explained_variance_ratio_

# Step 3: Compute the cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
reduced_embeddings_tsne = tsne.fit_transform(torch_embeddings)

# Plot the t-SNE reduced embeddings
plt.figure(figsize=(10, 6))
for source, color in zip(['Fox News', 'CNN'], ['red', 'blue']):
    indices = [i for i, label in enumerate(labels) if label == source]
    plt.scatter(reduced_embeddings_tsne[indices, 0], reduced_embeddings_tsne[indices, 1], label=source, alpha=0.6, color=color)

plt.title("t-SNE of Article Embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.savefig("tsne_plot.png", format="png", dpi=300)
plt.show()
# embedding_tensor = torch.tensor(emd["embedding"])
# print(embedding_tensor)