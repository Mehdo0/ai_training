import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torchvision.transforms as T

# THE AI BRAIN
class QuickDrawBrain(nn.Module):
    def __init__(self):
        super(QuickDrawBrain, self).__init__()
        
        # BLOCK 1: First set of magnifying glasses (Looks for basic edges/curves)
        # We upgraded out_channels to 32 (32 different magnifying glasses)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # BLOCK 2: Second set of magnifying glasses (Combines edges into complex shapes)
        # Takes the 32 channels from Block 1 and uses 64 new magnifying glasses
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # DROPOUT: Randomly turns off 25% of brain connections during training 
        # to force the AI to build stronger, more generalized pathways
        self.dropout = nn.Dropout(0.25)
        
        # BLOCK 3: The Decision Makers
        # Math Check: Our 28x28 image went through pool1 (shrunk to 14x14) 
        # and pool2 (shrunk to 7x7). 
        # 64 channels * 7 height * 7 width = 3136 flattened pixels
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=512) # A deep hidden layer
        self.fc2 = nn.Linear(in_features=512, out_features=35)         # Final output for 35 categories

    def forward(self, x):
        # Pass through Block 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Pass through Block 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten the 3D grid into a 1D line
        x = torch.flatten(x, 1) 
        
        # Pass through the new deep layers with Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# DATA LOADING & PREP
print("Loading NPY_FILE from the 'NPY_FILE/' folder...")

X_train = np.load('NPY_FILE/X_train.npy')
y_train = np.load('NPY_FILE/y_train.npy')
X_val = np.load('NPY_FILE/X_val.npy')
y_val = np.load('NPY_FILE/y_val.npy')

print("Converting to PyTorch Tensors and normalizing...")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# TRAINING SETUP
model = QuickDrawBrain()
criterion = nn.CrossEntropyLoss()           
optimizer = optim.Adam(model.parameters(), lr=0.001) 
epochs = 5

# Data Augmentation: The tilt tool
tilt_tool = T.RandomRotation(degrees=15)

print(f"\nStarting training for {epochs} epochs...")

# THE TRAINING LOOP
for epoch in range(epochs):
    model.train() 
    running_loss = 0.0
    
    for images, labels in train_loader:
        images = tilt_tool(images)      # Slight tilt to the image on the fly!
        optimizer.zero_grad()           
        outputs = model(images)         
        loss = criterion(outputs, labels) 
        loss.backward()                 
        optimizer.step()                
        
        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] complete. Average Loss: {avg_loss:.4f}")

# TESTING & EXPORT

print("\nTraining finished! Testing on hidden validation set...")

model.eval() 
correct_guesses = 0
total_drawings = len(y_val_tensor)

with torch.no_grad():
    test_outputs = model(X_val_tensor)
    _, predictions = torch.max(test_outputs, 1)
    correct_guesses = (predictions == y_val_tensor).sum().item()

accuracy = (correct_guesses / total_drawings) * 100
print(f"Final Validation Accuracy: {accuracy:.2f}%")

save_path = 'trained_brain.pth'
torch.save(model.state_dict(), save_path)

print(f"\nSuccess! The trained AI has been saved as '{save_path}'.")
print("It is now ready to be plugged into your game server!")