from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print(device)

# Download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# Split into a training set and a test set using a stratified k fold
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

# move data to gpu
X_train_torch = torch.tensor(X_train, dtype=torch.float32, device=device)
X_test_torch = torch.tensor(X_test, dtype=torch.float32, device=device)

# time pca calculation
torch_start = time.time()

# Center data
mean = torch.mean(X_train_torch, dim=0)  #! torch method
X_train_torch -= mean
X_test_torch -= mean

# Eigen-decomposition
U, S, V = torch.linalg.svd(X_train_torch, full_matrices=False)  #! torch method
components = V[:n_components]
eigenfaces = components.reshape((n_components, h, w))

# project into PCA subspace
X_transformed = torch.matmul(X_train_torch, components.T)  #! torch method
print(X_transformed.shape)
X_test_transformed = torch.matmul(X_test_torch, components.T)  #! torch method
print(X_test_transformed.shape)

torch_end = time.time()
torch_duration = torch_end - torch_start
print(f"Duration to find PCA of eigenfaces using Torch: {torch_duration:.6f}")


# Qualitative evaluation of the predictions using matplotlib
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col + 1, 2.4 * n_row + 1))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

    plt.savefig("figs/eigenfaces.png")
    plt.clf()


eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]


def compactness():
    # evaluate the performance of the dimensionality reduction via a compactness plot
    explained_variance = (S**2) / (n_samples - 1)
    total_var = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_var
    ratio_cumsum = np.cumsum(explained_variance_ratio)
    print(ratio_cumsum.shape)
    eigenvalueCount = np.arange(n_components)

    plt.figure(figsize=(10, 6))
    plt.plot(eigenvalueCount, ratio_cumsum[:n_components])
    plt.title("Compactness")
    plt.savefig("figs/compactness.png")
    plt.clf()


def random_forest():
    # build random forest
    estimator = RandomForestClassifier(n_estimators=150, max_depth=15, max_features=150)
    estimator.fit(X_transformed, y_train)  # expects X as (n_samples, n_features)

    predictions = estimator.predict(X_test_transformed)
    correct = predictions == y_test
    total_test = len(X_test_transformed)
    # print("Gnd Truth:", y_test)
    print("Total Testing", total_test)
    print("Predictions", predictions)
    print("Which Correct:", correct)
    print("Total Correct:", np.sum(correct))
    print("Accuracy:", np.sum(correct) / total_test)

    print(classification_report(y_test, predictions, target_names=target_names))
