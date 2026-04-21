import numpy as np
from sklearn.model_selection import train_test_split
import os

#array of every distinc drawing file
categories = ['airplane', 'angel', 'anvil', 'bear', 'bee', 'brain', 'butterfly', 'camera', 'castle', 'cat', 'clarinet', 'computer', 'crab', 'dog', 'Eiffle Tower', 'elephant', 'fish', 'frog', 'guitar', 'hamburger', 'hand', 'horse', 'lion', 'mouse', 'owl', 'pineapple', 'radio', 'rifle', 'snowman', 'steak', 'suitecase', 'sword', 'toilet', 'train', 'tree']

#Limit the items so you don't overwhelm your local processing power
max_items_per_class = 10000

# Arrays to hold our merged data
# Quick Draw bitmaps are 28x28 images flattened into an array of 784 pixels
X = np.empty((0, 784), dtype=np.uint8) 
y = np.empty((0,), dtype=int)

for idx, category in enumerate(categories):
    # Load the numpy file for the specific category
    file_path = f'NPY_FILE/{category}.npy'
    data = np.load(file_path)
    
    # Grab only the subset to save memory
    data = data[:max_items_per_class] 
    
    # Append the drawing data to X
    X = np.concatenate((X, data), axis=0)
    
    # Create labels for this category (e.g., 'apple' is 0, 'bowtie' is 1) and append to y
    labels = np.full(data.shape[0], idx)
    y = np.concatenate((y, labels), axis=0)
    
    print(f"Loaded {len(data)} drawings for {category} (Label: {idx})")


# Apply the 80/20 rule (Train/Validation split)
# train_test_split automatically shuffles the data so the AI doesn't learn in order
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

# Save the processed data so it's ready for PyTorch
np.save('./NPY_FILE/X_train.npy', X_train)
np.save('./NPY_FILE/X_val.npy', X_val)
np.save('./NPY_FILE/y_train.npy', y_train)
np.save('./NPY_FILE/y_val.npy', y_val)
