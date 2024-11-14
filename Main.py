import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import numpy as np

with open("Contrastive with Temporal Shuffling.py") as f:
    code = f.read()
    exec(code)

class SSLEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):  
        super(SSLEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x.mean(dim=-1)  # Global average pooling for representation

# Initialize the model and load the state dict
ssl_encoder = torch.load("Contrastive_with_Temporal_Shuffling.pth")

for param in ssl_encoder.parameters():
    param.requires_grad = False

# Load the UWaveGestureLibrary dataset and convert to float32
uwave_data = torch.load("Gesture/train.pt")
X_train = uwave_data['samples'].float()  # Convert to float32
y_train = uwave_data['labels']


x_test_data=torch.load("Gesture/test.pt")
X_test=x_test_data['samples'].float() # Convert to float32
y_test=x_test_data['labels']

print("X_test shape:", X_test.shape)
print("X_train shape:", X_train.shape)
# Extracting embeddings using the SSL encoder
with torch.no_grad():
    X_embeddings = ssl_encoder(X_train).numpy()  # Convert to numpy if needed for sklearn

# Reshaping the embeddings to 2D
X_embeddings = X_embeddings.reshape(X_embeddings.shape[0], -1)

# Standardize embeddings
scaler = StandardScaler()
X_embeddings = scaler.fit_transform(X_embeddings)

with torch.no_grad():
    X_test_embeddings = ssl_encoder(X_test).numpy()
X_test_embeddings = X_test_embeddings.reshape(X_test_embeddings.shape[0], -1)
X_test_embeddings = scaler.transform(X_test_embeddings)


# Training kNN classifier with k=5
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_embeddings, y_train)

# Evaluate model
y_pred = knn_classifier.predict(X_test_embeddings)
print(classification_report(y_test, y_pred))
