import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

# Read the dataset
df = pd.read_csv("dataset.csv")

# Fill missing values with an empty string
df['text'].fillna('', inplace=True)

# Preprocess text data
vectorizer = TfidfVectorizer(stop_words='english')
X_text = vectorizer.fit_transform(df['text']).toarray()

# Convert y to integers (if necessary)
df['label'] = df['label'].apply(lambda x: 0 if x == 'fake' else 1)

# Convert y to a PyTorch tensor
y = torch.LongTensor(df['label'].values)

# Convert text features to tensors
X_text = torch.FloatTensor(X_text)

# Load and preprocess image data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert single-channel image to RGB
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to extract image metadata
def extract_image_metadata(url):
    try:
        response = requests.head(url)
        response.raise_for_status()  # Raise an exception for non-200 status codes
        content_type = response.headers.get('content-type')
        if content_type.startswith('image'):
            image = Image.open(BytesIO(requests.get(url).content))
            width, height = image.size
            return [width, height]
        else:
            print(f"Invalid content type for URL: {url}")
    except requests.RequestException as e:
        print(f"Failed to fetch image metadata from URL: {url}, Error: {e}")
    except Exception as e:
        print(f"Error processing image from URL: {url}, Error: {e}")
    return [0, 0]  # Default dimensions if metadata extraction fails

image_metadata = df['top_img'].apply(lambda url: extract_image_metadata(url))

# Combine text and image data
X_combined = torch.cat((X_text, torch.FloatTensor(image_metadata)), dim=1)

# Prepare adjacency matrix based on TF-IDF weighted word co-occurrence
adjacency_matrix = np.dot(X_combined, X_combined.T)
adjacency_matrix = torch.FloatTensor(adjacency_matrix)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Define the number of most significant connections per node
k = 10

# Calculate the cosine similarity between the features of the nodes
similarity_matrix = torch.cosine_similarity(X_train, X_train)

# Get the top k indices
top_k_indices = torch.topk(similarity_matrix, k=k, dim=0)[1]

# Repeat the top k indices along dimension 0
top_k_indices_repeated = top_k_indices.repeat_interleave(top_k_indices.size(0), dim=0)

# Concatenate the repeated indices along dimension 1
edge_index = torch.cat([top_k_indices_repeated, top_k_indices_repeated.t()], dim=0)

# Get the source and target nodes of the edges
source_nodes = edge_index[0]
target_nodes = edge_index[1]

# Remove self-loops from the edge_index
edge_index = edge_index[:, source_nodes != target_nodes]

# Initialize edge_index differently based on whether it is empty or not
if edge_index.numel() == 0:  # Check if edge_index is empty
    edge_index = torch.zeros((2, 2), dtype=torch.long)
else:
    edge_index = torch.cat([edge_index, edge_index.new_zeros(edge_index.size(0), 2).long()], dim=0)
    edge_index[edge_index[:, 0] == edge_index[:, 1], 1] = edge_index.size(1)
    edge_index = edge_index[edge_index[:, 1] > edge_index[:, 0], :]

class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.conv4 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv4(x, edge_index)
        return F.log_softmax(x, dim=1)

# Instantiate the model
input_dim = X_combined.shape[1]
hidden_dim = 128
output_dim = 2  # Two classes: Real and Fake
model = GCNModel(input_dim, hidden_dim, output_dim)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate

# Train-validation-test split
X_train_val, X_test, y_train_val, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Convert X_train, X_val, and X_test to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.LongTensor(y_val)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# Wrap the data in PyTorch Geometric Data objects
train_data = Data(x=X_train, edge_index=edge_index, y=y_train)
val_data = Data(x=X_val, edge_index=edge_index, y=y_val)
test_data = Data(x=X_test, edge_index=edge_index, y=y_test)

# Define the size of the train and validation masks
num_nodes_train = X_train.size(0)
num_nodes_val = X_val.size(0)

# Create the train and validation masks
train_mask = torch.zeros(num_nodes_train, dtype=torch.bool)
val_mask = torch.zeros(num_nodes_val, dtype=torch.bool)
train_mask[:num_nodes_train // 2] = True  # Assuming first half of nodes are in the training set
val_mask[:num_nodes_val // 2] = True  # Assuming first half of nodes are in the validation set

# Assign the train and validation masks to train and validation data
train_data.train_mask = train_mask
val_data.val_mask = val_mask

# Move the model and data to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
train_data.to(device)
val_data.to(device)
test_data.to(device)

# Define the training loop
def train(model, train_data, criterion, optimizer, device):
    model.train()
    optimizer.zero_grad()
    out = model(train_data.x, train_data.edge_index)
    loss = criterion(out[train_data.train_mask], train_data.y[train_data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Define the accuracy function
def accuracy(pred, y):
    _, predicted = torch.max(pred, 1)
    correct = (predicted == y).sum().item()
    total = y.size(0)
    return correct / total

# Define the evaluation function for the test set
def test(model, test_data, criterion):
    model.eval()
    with torch.no_grad():
        out = model(test_data.x, test_data.edge_index)
        loss = criterion(out[test_data.y], test_data.y)
        acc = accuracy(out[test_data.y], test_data.y)
    return loss.item(), acc

# Train the model
best_val_loss = float('inf')
best_model = None
for epoch in range(100):
    train_loss = train(model, train_data, criterion, optimizer, device)
    val_loss, val_acc = test(model, val_data, criterion)
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

    # Check if the current validation loss is the best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

# Evaluate the best model on the test set
test_loss, test_acc = test(best_model, test_data, criterion)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')