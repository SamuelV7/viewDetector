import torch
import ollama

# Generate some example data

# print(X_pca)
model = "llama3.2"

emd = ollama.embeddings(
    model=model, 
    prompt="Howdy how is life"
)

embedding_tensor = torch.tensor(emd["embedding"])
print(embedding_tensor)