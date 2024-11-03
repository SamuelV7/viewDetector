from dataclasses import dataclass
import torch
import ollama
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import mplcursors  # Import mplcursors for hover functionality
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import os
from sentence_transformers.quantization import quantize_embeddings
from sklearn.preprocessing import StandardScaler

# Generate some example data

model = "llama3.2"
# model = "gemma2:27b"

# Read JSON file
def read_data_from_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def load_data(file_name):
    the_file = read_data_from_json(file_name)
    name = the_file["name"]
    source_name = the_file["article"]
    content = [x["content"] for x in source_name]
    return name, content

@dataclass
class Source:
    list_of_content: list
    name: str

def load_sources():
    # get all json files from directory
    path = "/workspace/spectra/data/"
    json_files = [f for f in os.listdir("/workspace/spectra/data/") if f.endswith(".json")]
    data = [load_data(f"{path}/{f}") for f in json_files]
    return [Source(content, source) for source, content in data]

def ollama_embed(sources: list[Source]):
    print("Starting to make embeddings")
# Create embeddings using ollama
    # tensor list 
    tensor_list = []
    labels = []
    for source in sources:
        for content in source.list_of_content:
            embed = torch.tensor(ollama.embeddings(model, prompt=content)["embedding"])
            tensor_list.append(embed)
        labels += [source.name] * len(source.list_of_content)
    print("Done making embeddings")
    return torch.stack(tensor_list), labels

def mixedbread_embed(cnn_content, fox_news_content):
    # Load the SentenceTransformer model
    dimensions = 1024
    # 2. load model
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=dimensions, device='cpu')
    # The prompt used for query retrieval tasks:
    # query_prompt = 'Represent this sentence for searching relevant passages: '
    # 2. Encode
    cnn_embeddings = model.encode(cnn_content)
    fox_news_embeddings = model.encode(fox_news_content)
    # binary_cnn = quantize_embeddings(cnn_embeddings, precision)
    # binary_fox_news = quantize_embeddings(fox_news_embeddings)
    print("Done making embeddings")
    return cnn_embeddings,fox_news_embeddings

source_file = load_sources()

tensors_embed, labels = ollama_embed(source_file)

# Perform PCA to reduce dimensions
torch_embeddings = torch.tensor(tensors_embed, device='cpu')
# normalize embeddings with sklearn
torch_embeddings = StandardScaler().fit_transform(torch_embeddings)
pca = PCA(n_components=3)
reduced_embedding = pca.fit_transform(torch_embeddings)

# Transform the data to 2 dimensions
# reduced_embeddings = pca.transform(torch_embeddings)[:, :2]

# 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define unique labels and colors
unique_labels = list(set(labels))
colors = plt.cm.tab20.colors  # Use a colormap for distinct colors
color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
point_colors = [color_map[label] for label in labels]

# 3D scatter plot
scatter = ax.scatter(
    reduced_embedding[:, 0],
    reduced_embedding[:, 1],
    reduced_embedding[:, 2],
    c=point_colors,
    alpha=0.6,
    edgecolor="k"
)

# Hover functionality with mplcursors
cursor = mplcursors.cursor(scatter, hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f"{labels[sel.index]}"))

# Plot configuration
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
plt.title("3D PCA Embedding Visualization by Source")

# Create custom legend
for label in unique_labels:
    ax.scatter([], [], [], color=color_map[label], label=label)
plt.legend(title="Sources")
plt.show()
# plt.figure(figsize=(10, 8))
# unique_labels = list(set(labels))
# colors = plt.cm.tab20.colors  # Use a colormap with diverse colors

# # Create a color mapping for each unique label
# color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
# point_colors = [color_map[label] for label in labels]

# # Scatter plot with points colored by label
# scatter = plt.scatter(
#     reduced_embeddings[:, 0],
#     reduced_embeddings[:, 1],
#     c=point_colors,
#     alpha=0.6,
#     edgecolor="k"
# )

# # Add labels for hover functionality
# cursor = mplcursors.cursor(scatter, hover=True)
# cursor.connect(
#     "add",
#     lambda sel: sel.annotation.set_text(f"{labels[sel.index]}")
# )

# # Plot configuration
# plt.xlabel("PCA Component 1")
# plt.ylabel("PCA Component 2")
# plt.title("Embedding Visualization by Source")
# plt.grid(True)

# # Create a custom legend to match colors with labels
# for label in unique_labels:
#     plt.scatter([], [], color=color_map[label], label=label)
# plt.legend(title="Sources")

# plt.show()
