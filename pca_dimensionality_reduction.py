from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load dataset (example: Iris dataset)
iris = load_iris()
X = iris.data

# Perform PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(f'Original shape: {X.shape}')
print(f'Reduced shape: {X_reduced.shape}')
