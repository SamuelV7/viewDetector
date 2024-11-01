import torch

# Generate some example data
X = torch.randn(100, 50)  # 100 samples, 50 features

# Center the data
X_centered = X - X.mean(dim=0)

# Perform SVD
U, S, V = torch.svd(X_centered)

# Project onto the first k principal components
k = 10
X_pca = torch.matmul(X_centered, V[:, :k])

print(X_pca)