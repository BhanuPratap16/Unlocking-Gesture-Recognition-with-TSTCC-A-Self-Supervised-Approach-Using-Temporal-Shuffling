# !pip install torch
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
model_data = torch.load("Gesture/train.pt")
X_train = model_data['samples']
y_train = model_data['labels']

pre_data_train = torch.load('HAR/train.pt')
pre_data_test = torch.load('HAR/test.pt')
pre_X_train = pre_data_train['samples']
pre_y_train = pre_data_train['labels']
pre_X_test = pre_data_test['samples']
pre_y_test = pre_data_test['labels']

# Dataset and DataLoader
class GestureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = self.features[idx]
        label = self.labels[idx]
        return {"features": torch.tensor(sample, dtype=torch.float32), 
                "labels": torch.tensor(label, dtype=torch.long)}

train_dataset = GestureDataset(pre_X_train, pre_y_train)
test_dataset = GestureDataset(pre_X_test, pre_y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Data Augmentation
def jitter(data, sigma=0.03):
    return data + np.random.normal(loc=0, scale=sigma, size=data.shape)

def scaling(data, sigma=0.1):
    factor = np.random.normal(loc=1.0, scale=sigma, size=(data.shape[1], 1))
    return data * factor

def shuffle_segments(data, segment_size=5):
    segments = [data[i:i+segment_size] for i in range(0, len(data), segment_size)]
    np.random.shuffle(segments)
    return np.concatenate(segments)

# Model Definition
class TSTCCEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TSTCCEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x.mean(dim=-1)

class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim, projection_dim):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, projection_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Contrastive Loss
def contrastive_loss(z_i, z_j, temperature=0.5):
    z_i = nn.functional.normalize(z_i, dim=1)
    z_j = nn.functional.normalize(z_j, dim=1)
    similarity_matrix = torch.mm(z_i, z_j.T) / temperature
    labels = torch.arange(z_i.size(0)).to(z_i.device)
    return nn.functional.cross_entropy(similarity_matrix, labels)

# Initialize model, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = TSTCCEncoder(input_dim=3, hidden_dim=128).to(device)
projection_head = ProjectionHead(hidden_dim=128, projection_dim=64).to(device)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projection_head.parameters()), lr=1e-3)

# Training Loop
for epoch in range(10):
    for batch in train_loader:
        original_data = batch['features'].to(device).float()
        aug1_data = jitter(original_data).float()
        aug2_data = scaling(original_data).float()
        shuffled_data = shuffle_segments(original_data.cpu().numpy())
        shuffled_data = torch.tensor(shuffled_data, dtype=torch.float32).to(device)

        # Contrastive Task
        z_i = projection_head(encoder(aug1_data))
        z_j = projection_head(encoder(aug2_data))
        loss_contrastive = contrastive_loss(z_i, z_j)

        # Shuffling Task
        z_original = encoder(original_data)
        z_shuffled = encoder(shuffled_data)
        loss_shuffling = torch.mean((z_original - z_shuffled) ** 2)

        loss = loss_contrastive + loss_shuffling
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Evaluation
torch.save(encoder,"Contrastive_with_Temporal_Shuffling.pth")
encoder.eval()
test_features = []
test_labels = []

with torch.no_grad():
    for batch in test_loader:
        data = batch['features'].to(device).float()
        labels = batch['labels'].to(device)
        features = encoder(data)
        test_features.append(features)
        test_labels.append(labels)

test_features = torch.cat(test_features).cpu().numpy()
test_labels = torch.cat(test_labels).cpu().numpy()

# Logistic Regression
classifier = LogisticRegression(max_iter=1000)
classifier.fit(test_features, test_labels)
y_pred = classifier.predict(test_features)
logistic_accuracy = accuracy_score(test_labels, y_pred)
print(f"Logistic Regression Test Accuracy: {logistic_accuracy * 100:.2f}%")

# k-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(test_features, test_labels)
y_pred_knn = knn.predict(test_features)
knn_accuracy = accuracy_score(test_labels, y_pred_knn)
print(f"KNN Test Accuracy: {knn_accuracy * 100:.2f}%")
