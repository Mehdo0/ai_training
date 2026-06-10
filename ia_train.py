import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torchvision.transforms as T
from google.colab import drive
drive.mount('/content/drive')

# ==========================================
# THE HEAVILY UPGRADED AI BRAIN
# ==========================================
class QuickDrawBrain(nn.Module):
    def __init__(self):
        super(QuickDrawBrain, self).__init__()

        # BLOCK 1: 28x28 -> 14x14
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32) # Stabilizes math outputting from conv1_1
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # BLOCK 2: 14x14 -> 7x7
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # BLOCK 3: Deep Decision Makers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.4) # Increased to 40% to prevent memorization
        self.fc2 = nn.Linear(512, 35)   # Output layer for 35 classes (FIXED FROM 34 TO 35)

    def forward(self, x):
        # Block 1 forward
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)

        # Block 2 forward
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)

        # Flatten and Decision making
        x = torch.flatten(x, 1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==========================================
# DATA LOADING & PREP (GOOGLE DRIVE PATHS)
# ==========================================
print("Loading data from Google Drive...")
X_train = np.load('/content/drive/MyDrive/NPY_FILES/X_train.npy')
y_train = np.load('/content/drive/MyDrive/NPY_FILES/y_train.npy')
X_val = np.load('/content/drive/MyDrive/NPY_FILES/X_val.npy')
y_val = np.load('/content/drive/MyDrive/NPY_FILES/y_val.npy')

print("Converting to PyTorch Tensors and normalizing...")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True) # Increased batch size for faster GPU training

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# ==========================================
# TRAINING SETUP
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# Diagnostic: Print the actual number of classes from training data
actual_num_classes = len(torch.unique(y_train_tensor))
print(f"Actual number of classes found in training data: {actual_num_classes}")

model = QuickDrawBrain().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# LEARNING RATE SCHEDULER: Drops the learning rate by 50% if the validation loss
# doesn't improve for 2 epochs straight. This helps the AI fine-tune.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Increased epochs! Let's push it to 15 epochs since Colab handles it fast.
epochs = 14
tilt_tool = T.RandomRotation(degrees=15)

# ==========================================
# THE TRAINING & TESTING LOOP
# ==========================================
print(f"\nStarting training for {epochs} epochs...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = tilt_tool(images)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Run a quick validation check at the end of every epoch to update the scheduler
    model.eval()
    val_loss = 0.0
    correct_guesses = 0
    total_drawings = len(y_val_tensor)

    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            loss = criterion(val_outputs, val_labels)
            val_loss += loss.item()

            _, predictions = torch.max(val_outputs, 1)
            correct_guesses += (predictions == val_labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    accuracy = (correct_guesses / total_drawings) * 100

    # Let the scheduler look at the validation loss
    scheduler.step(avg_val_loss)

    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {accuracy:.2f}%")

# Save directly to your Google Drive
save_path = '/content/drive/MyDrive/NPY_FILES/trained_brain.pth'
torch.save(model.state_dict(), save_path)
print(f"\nSuccess! The upgraded AI has been saved directly to your Google Drive.")