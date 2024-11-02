import torch
import ollama
import json
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


# make embeddings then do pca on them
cnn_embeddings = [ollama.embeddings(model, x) for x in cnn_content]
fox_news_embeddings = [ollama.embeddings(model, x) for x in fox_news_content]

# make embeddings one list
embeddings = cnn_embeddings + fox_news_embeddings
assert len(embeddings) == len(cnn_embeddings) + len(fox_news_embeddings)

# do pca on embeddings
torch_embeddings = torch.tensor(embeddings)
pca = torch.pca_lowrank(torch_embeddings, q=2)


# embedding_tensor = torch.tensor(emd["embedding"])
# print(embedding_tensor)