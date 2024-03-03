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

# Read the dataset
df = pd.read_csv("news_articles.csv")

# Fill missing values with an empty string
df['text_without_stopwords'].fillna('', inplace=True)

# Preprocess text data
vectorizer = TfidfVectorizer(stop_words='english')
X_text = vectorizer.fit_transform(df['text_without_stopwords']).toarray()

# Convert y to a 1D numpy array
y = df['label'].apply(lambda x: 0 if x == 'Fake' else 1).values

# Convert y to PyTorch tensor
y = torch.LongTensor(y)

# Convert text features to tensors
X_text = torch.FloatTensor(X_text)

# Prepare adjacency matrix based on TF-IDF weighted word co-occurrence
adjacency_matrix = np.dot(X_text, X_text.T)
adjacency_matrix = torch.FloatTensor(adjacency_matrix)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

# Define the number of most significant connections per node
k = 10

# Calculate the cosine similarity between the features of the nodes
similarity_matrix = torch.cosine_similarity(X_train, X_train)

print("Similarity Matrix Shape:", similarity_matrix.shape)

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

print("Edge Index Size:", edge_index.size())

# Define the Graph Convolutional Network (GCN) model
class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

# Instantiate the model
input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = 2  # Two classes: Real and Fake
model = GCNModel(input_dim, hidden_dim, output_dim)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate

# Convert X_train and y_train to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)

# Wrap the data in a PyTorch Geometric Data object
train_data = Data(x=X_train, edge_index=edge_index, y=y_train)

# Define the size of the train mask
num_nodes = X_train.size(0)

# Create the train mask
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[:num_nodes // 2] = True  # Assuming first half of nodes are in the training set

# Assign the train mask to train_data
train_data.train_mask = train_mask


# Define the training loop
def train(model, train_data, criterion, optimizer, device):
    model.train()
    optimizer.zero_grad()
    out = model(train_data.x, train_data.edge_index)
    loss = criterion(out[train_data.train_mask], train_data.y[train_data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
train_data.to(device)

# Define the size of the test mask
num_nodes = X_train.size(0)

# Create the test mask
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask[num_nodes // 2:] = True  # Assuming the second half of nodes are in the test set

# Assign the test mask to train_data
train_data.test_mask = test_mask


# Train the model for 100 epochs
for epoch in range(100):
    loss = train(model, train_data, criterion, optimizer, device)
    print(f'Epoch: {epoch+1}, Loss: {loss:.4f}')

def accuracy(pred, y):
    correct = torch.sum(torch.argmax(pred, dim=1) == y).item()
    acc = correct / y.shape[0]
    return acc

# Define the evaluation function
def evaluate(model, data, criterion):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.test_mask], data.y[data.test_mask])
        acc = accuracy(out[data.test_mask], data.y[data.test_mask])
    return loss, acc

# Evaluate the model on the test set
loss, acc = evaluate(model, train_data, criterion)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')