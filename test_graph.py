# Import necessary libraries (assuming all imports are relevant)
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import requests
from io import BytesIO
import networkx as nx
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torchvision import transforms
from torch_geometric.utils.convert import to_networkx
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import requests
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Function to extract image metadata
def extract_image_metadata(url):
    try:
        response = requests.head(url)
        response.raise_for_status()  # Raise an exception for non-200 status codes
        
        # Extract image content type and size
        content_type = response.headers.get('content-type')
        content_length = int(response.headers.get('content-length', 0))
        
        if content_type.startswith('image'):
            # Fetch the image
            image_data = requests.get(url).content
            image = Image.open(BytesIO(image_data))
            
            # Extract image dimensions
            width, height = image.size
            
            # Calculate aspect ratio
            aspect_ratio = width / height
            
            # Calculate image resolution
            resolution = width * height
            
            # Return metadata
            metadata = {
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio,
                'resolution': resolution,
                'content_length': content_length,
            }
            print(metadata)
            return metadata
        else:
            print(f"Invalid content type for URL: {url}")
    except requests.RequestException as e:
        print(f"Failed to fetch image metadata from URL: {url}, Error: {e}")
    except Exception as e:
        print(f"Error processing image from URL: {url}, Error: {e}")
    return None

# Read the training dataset
train_df = pd.read_csv("/Users/sindhugunda/Documents/Information Technology/Year 3/CST3990/Final-Year-Project/traindata.csv")

# Read the testing dataset
test_df = pd.read_csv("/Users/sindhugunda/Documents/Information Technology/Year 3/CST3990/Final-Year-Project/multimodal_test_public.csv")

# Set the desired size for your subset
subset_size = 100

# Randomly sample data points from the DataFrame
train_df = train_df.sample(n=subset_size, random_state=42)
test_df = test_df.sample(n=subset_size, random_state=42)

# Preprocess image data and handle missing values for training
train_df = train_df.dropna(subset=['image_url'])
train_image_metadata = train_df['image_url'].apply(lambda url: extract_image_metadata(url)).dropna()
train_image_metadata_df = pd.DataFrame(train_image_metadata.tolist())

# Preprocess text data for training
train_df['clean_title'].fillna('', inplace=True)
train_df['title'].fillna('', inplace=True)
# TF-IDF 
# TF - term frequency
# IDF - Inverse document frequency measures the importance of the word in the document 
# it is thre ration of the no of documents to the number of documents containing the term
# TF-IDF basically combines both and gives a score for each term in the document, all of it is in a matrix 
vectorizer = TfidfVectorizer(stop_words='english')
X_text_train = vectorizer.fit_transform(train_df['clean_title'] + " " + train_df['title']).toarray()
# Randomly sample data points from X_text_train to match the size of train_image_metadata_df
X_text_train = X_text_train[:train_image_metadata_df.shape[0]]

# Randomly sample data points from train_image_metadata_df to match the size of X_text_train
train_image_metadata_df = train_image_metadata_df.sample(n=X_text_train.shape[0], random_state=42)

# Combine text features and image metadata for training
X_combined_train = np.concatenate((X_text_train, train_image_metadata_df.to_numpy()), axis=1)

# Convert y to integers for training
train_df['2_way_label'] = train_df['2_way_label'].apply(lambda x: 1 if x == 'fake' else 0 if x == 'real' else x)
y_train = train_df['2_way_label'].values

# Preprocess image data and handle missing values for testing
test_df = test_df.dropna(subset=['image_url'])
test_image_metadata = test_df['image_url'].apply(lambda url: extract_image_metadata(url)).dropna()
test_image_metadata_df = pd.DataFrame(test_image_metadata.tolist())

# Preprocess text data for testing
test_df['clean_title'].fillna('', inplace=True)
test_df['title'].fillna('', inplace=True)
X_text_test = vectorizer.transform(test_df['clean_title'] + " " + test_df['title']).toarray()

# Pad test_image_metadata_df array to match the number of rows in X_text_test
# this is to match the shapes of the images array and text array
num_rows_diff = X_text_test.shape[0] - test_image_metadata_df.shape[0]
if num_rows_diff > 0:
    padding = np.zeros((num_rows_diff, test_image_metadata_df.shape[1]))  # Create padding with zeros
    test_image_metadata_padded = np.concatenate((test_image_metadata_df.to_numpy(), padding), axis=0)
else:
    test_image_metadata_padded = test_image_metadata_df.to_numpy()

# Check the shapes of the arrays
print("Shape of X_text_test:", X_text_test.shape)
print("Shape of test_image_metadata_df:", test_image_metadata_df.to_numpy().shape)

# Combine text features and image metadata for testing
X_combined_test = np.concatenate((X_text_test, test_image_metadata_padded), axis=1)

# Convert y to integers for testing
test_df['2_way_label'] = test_df['2_way_label'].apply(lambda x: 1 if x == 'fake' else 0 if x == 'real' else x)
y_test = test_df['2_way_label'].values

# Convert features and labels to tensors
X_train_tensor = torch.FloatTensor(X_combined_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_combined_test)
y_test_tensor = torch.LongTensor(y_test)

# Construct the edge index here based on your data
# Example: Constructing edge index using a similarity-based approach
# Your code to construct edge index goes here...

import random

# # Function to randomly remove edges from the edge index
# def remove_random_edges(edge_index, removal_percentage=0.2):
#     num_edges = edge_index.size(1)
#     num_edges_to_remove = int(removal_percentage * num_edges)
#     edges_to_remove = random.sample(range(num_edges), num_edges_to_remove)
#     pruned_edge_index = edge_index[:, ~torch.tensor([i in edges_to_remove for i in range(num_edges)])]
#     return pruned_edge_index

# Example usage
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)  # Example edge index
print("Original Edge Index:")
print(edge_index)

# edge_index = remove_random_edges(edge_index, removal_percentage=0.2)
# print("\nPruned Edge Index (20% edges removed randomly):")
# print(edge_index)


# Create edge index for a fully connected graph
num_nodes = X_combined_train.shape[0]
edge_index = torch.tensor(list(itertools.combinations(range(num_nodes), 2)), dtype=torch.long).t().contiguous()

# Remove self-loops from the edge_index
edge_index = torch_geometric.utils.remove_self_loops(edge_index)[0]

# Ensure edge_index is on the correct device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
edge_index = edge_index.to(device)

# Ensure edge_index is on the correct device
edge_index = edge_index.to(device)

# Create the Data object with x (features) and edge_index attributes
data = Data(x=torch.FloatTensor(X_combined_train), edge_index=edge_index)

# Convert to NetworkX graph
graph = to_networkx(data, to_undirected=True)

# Convert NetworkX graph to edge list
edge_list = list(nx.to_edgelist(graph))

# Extract source and target nodes from the edge list
source_nodes = [edge[0] for edge in edge_list]
target_nodes = [edge[1] for edge in edge_list]


# Define the GCN model
class SimpleGCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(SimpleGCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    
# Instantiate the model
input_dim = X_combined_train.shape[1]  # Use X_combined_train.shape[1] for the input dimension
hidden_dim = 64
output_dim = 2
dropout = 0.5
model = SimpleGCNModel(input_dim, hidden_dim, output_dim, dropout)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Determine the total number of nodes in the graph
num_nodes = edge_index.max().item() + 1

# Create masks with correct length
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(X_combined_test.shape[0], dtype=torch.bool)  # Use X_combined_test.shape[0] for the number of nodes

# Generate sequential node indices for training data
train_df['node_index'] = range(len(train_df))

# Generate sequential node indices for testing data
test_df['node_index'] = range(len(test_df))

# Convert NumPy array to PyTorch tensor
X_train_indices = torch.tensor(train_df['node_index'].values, dtype=torch.long)
X_test_indices = torch.tensor(test_df['node_index'].values, dtype=torch.long)

# Determine the total number of unique node indices in the training data
num_unique_nodes_train = train_df['node_index'].nunique()
num_unique_nodes_test = test_df['node_index'].nunique()

# Determine the total number of nodes in the graph
num_nodes = edge_index.max().item() + 1

# Create masks with correct length
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(X_combined_test.shape[0], dtype=torch.bool)

# Set the appropriate indices to True for the training and test sets
train_mask[:num_unique_nodes_train] = True
test_mask[:num_unique_nodes_test] = True

# Convert source and target nodes to PyTorch tensors
source_nodes = torch.tensor(source_nodes, dtype=torch.long)
target_nodes = torch.tensor(target_nodes, dtype=torch.long)

# Adjust the training and evaluation loops to handle data correctly
def train(model, X, y, edge_index, criterion, optimizer, device, mask):
    model.train()
    optimizer.zero_grad()
    out = model(X, edge_index)  # No need to send data to device here
    print("Shape of out tensor:", out.shape)
    print("Shape of mask tensor:", mask.shape)
    # Apply mask to output tensor
    masked_out = out[mask].view(-1, output_dim)
    print("Shape of masked_out tensor:", masked_out.shape)
    
    # Apply mask to y tensor
    masked_y = y[mask.nonzero(as_tuple=True)]
    
    loss = criterion(masked_out, masked_y)  # Apply mask
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, X, y, edge_index, criterion, device, mask):
    model.eval()
    with torch.no_grad():
        out = model(X.to(device), edge_index.to(device))
        masked_out = out[mask].view(-1, output_dim)
        masked_y = y[mask.nonzero(as_tuple=True)].squeeze()  # Adjusted to get non-zero indices
        loss = criterion(masked_out, masked_y)
        pred = masked_out.argmax(dim=1)
        y_true = masked_y.cpu().numpy()
        y_pred = pred.cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    return loss.item(), acc, precision, recall, f1

# Adjusted training and evaluation loops
for epoch in range(100):
    train_loss = train(model, X_train_tensor, y_train_tensor, edge_index, criterion, optimizer, device, train_mask)
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(model, X_test_tensor, y_test_tensor, edge_index, criterion, device, test_mask)
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}')

# Evaluation on test data
test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(model, X_test_tensor, y_test_tensor, edge_index, criterion, device, test_mask)
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}')

print(train_df['2_way_label'].unique())

# Count the occurrences of each label in the training and testing datasets
train_label_counts = train_df['2_way_label'].value_counts()
test_label_counts = test_df['2_way_label'].value_counts()

# Print the counts
print("Training Dataset:")
print("Fake rows:", train_label_counts.get(0, 0))  # Count occurrences of label 0 (fake)
print("Real rows:", train_label_counts.get(1, 0))  # Count occurrences of label 1 (real)
print("\nTesting Dataset:")
print("Fake rows:", test_label_counts.get(0, 0))   # Count occurrences of label 0 (fake)
print("Real rows:", test_label_counts.get(1, 0))   # Count occurrences of label 1 (real)

# Get model predictions
with torch.no_grad():
    out = model(X_test_tensor.to(device), edge_index.to(device))
    masked_out = out[test_mask].view(-1, output_dim)
    masked_y = y_test_tensor[test_mask.nonzero(as_tuple=True)].squeeze()  # Adjusted to get non-zero indices
    pred = masked_out.argmax(dim=1)
    y_true = masked_y.cpu().numpy()
    y_pred = pred.cpu().numpy()

# Define the labels
labels = ['Fake', 'Real']  # Reorder the labels to match the order in the confusion matrix

# Map numerical labels to string labels
y_true_labels = [labels[label] for label in y_true]
y_pred_labels = [labels[label] for label in y_pred]

# Calculate confusion matrix with specified labels
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)
print("Confusion Matrix", cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", cbar=False, 
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()