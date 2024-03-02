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
from torch_geometric.data import DataLoader, Batch

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

# Construct edge_index from adjacency_matrix
edge_index = torch.nonzero(adjacency_matrix, as_tuple=False).t()

# Define the Graph Convolutional Network (GCN) model
class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)  
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        edge_index = edge_index.long()  # Ensure edge_index is of type long
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

# Print the model architecture
print(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Convert data to PyTorch Geometric Data instances
train_data_list = [Data(x=x.unsqueeze(0), edge_index=edge_index, y=y) for x, y in zip(X_train, y_train)]
train_data = Batch.from_data_list(train_data_list)

test_data_list = [Data(x=x.unsqueeze(0), edge_index=edge_index, y=y) for x, y in zip(X_test, y_test)]
test_data = Batch.from_data_list(test_data_list)

# Train the model
def train_model(model, optimizer, criterion, train_data, adjacency_matrix, epochs=5):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_data.x, adjacency_matrix)
        loss = criterion(output, train_data.y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        
# Train the model
train_model(model, optimizer, criterion, train_data, adjacency_matrix)

# Evaluate the model
def evaluate_model(model, test_data, adjacency_matrix):
    model.eval()
    with torch.no_grad():
        output = model(test_data.x, adjacency_matrix)
        predicted = output.argmax(dim=1)
        accuracy = (predicted == test_data.y).sum().item() / len(test_data.y)
        print(f"Accuracy: {accuracy}")

# Evaluate the model
evaluate_model(model, test_data, adjacency_matrix)