import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as gnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import pandas as pd
import requests
from io import BytesIO

class ImageGNN(nn.Module):
    def __init__(self, image_feature_dim, text_feature_dim, hidden_dim):
        super(ImageGNN, self).__init__()
        # Define CNN for image processing
        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, image_feature_dim)
        
        # Define GNN for text processing
        self.text_gnn = gnn.GCNConv(text_feature_dim, hidden_dim)
        
        # Define fusion layer
        self.fusion_layer = nn.Linear(image_feature_dim + hidden_dim, 1)

    def forward(self, img_feats, text_feats, edge_index):
        # Image processing
        img_feats = self.cnn(img_feats)
        
        # Text processing
        text_feats = self.text_gnn(text_feats, edge_index)
        
        # Fusion
        combined_feats = torch.cat((img_feats, text_feats), dim=1)
        output = self.fusion_layer(combined_feats)
        return torch.sigmoid(output)

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        # Filter out rows with None values in either 'text' or 'images' column
        self.data = self.data.dropna(subset=['text', 'images']).reset_index(drop=True)

        # Filter out samples with None images
        self.data = self.data[self.data['images'].apply(lambda x: self._download_image(x) is not None)]

    def _download_image(self, image_url):
        try:
            # Download image from URL
            response = requests.get(image_url)
            response.raise_for_status()  # Raise an exception for non-200 status codes
            img = Image.open(BytesIO(response.content))
            return img
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image: {e}")
            return None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data.iloc[idx]
        text = entry['text']
        image_url = entry['images']
        
        img = self._download_image(image_url)
        
        if img is None:
            print(f"Skipping sample at index {idx} due to failed image download.")
            return None, text, None
        
        if self.transform:
            img = self.transform(img)
        
        label = entry.get('label', None)
        
        return img, text, label

# Define the transform for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to the same size expected by ResNet18
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB if the input image is grayscale
    transforms.ToTensor(),           # Convert PIL image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])

# Create custom dataset
dataset = CustomDataset('PolitiFact_fake_news_content.csv', transform=transform)

# Filter out None values from the dataset
filtered_dataset = []
for sample in dataset:
    if sample[0] is not None:
        filtered_dataset.append(sample)

# Create data loader
batch_size = 32# Create data loader from the filtered dataset
filtered_data_loader = DataLoader(filtered_dataset, batch_size=batch_size, shuffle=True)

# Create custom dataset
dataset = CustomDataset('PolitiFact_fake_news_content.csv', transform=transform)

# Example usage:
image_feature_dim = 512  # Dimension of image features from ResNet18
text_feature_dim = 128   # Dimension of text features
hidden_dim = 64          # Dimension of hidden layer in GNN

model = ImageGNN(image_feature_dim, text_feature_dim, hidden_dim)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary cross-entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
# Train the model using the filtered data loader
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for images, texts, labels in filtered_data_loader:
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(images, texts, torch_geometric.EdgeIndex)  # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize the model parameters
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(filtered_dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
