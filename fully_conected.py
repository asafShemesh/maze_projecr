import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score


# neural network structure
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def preprocess_and_extract_cells(image_path):
    """Preprocess the Sudoku image, extract the grid, and split it into cells."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours and extract the largest one
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contours found in {image_path}")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    grid = binary[y:y + h, x:x + w]
    grid = cv2.resize(grid, (450, 450))
    cells = []
    cell_size = 50 
    for i in range(9):
        for j in range(9):
            cell = grid[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            cell = cv2.resize(cell, (28, 28))
            cell = torch.tensor(cell, dtype=torch.float32).unsqueeze(0) / 255.0
            cells.append(cell)

    return cells

def recognize_and_print_numbers(cells, model):
    """Recognize numbers from extracted Sudoku cells using the trained model, and print the grid."""
    grid = []
    for cell in cells:
        with torch.no_grad():
            output = model(cell)
            _, predicted = torch.max(output, 1)
            predicted_digit = predicted.item()

            # If the model does not confidently predict a digit (i.e., predicted digit is 0), treat as empty
            if predicted_digit == 0:
                grid.append(0)
            else:
                grid.append(predicted_digit)
    
    # Print the 9x9 grid
    for i in range(9):
        print(grid[i*9:(i+1)*9])


def load_sudoku_dat_files(dat_folder):
    """Load the dat files representing Sudoku grids."""
    dat_files = [f for f in os.listdir(dat_folder) if f.endswith('.dat')]
    sudoku_data = []
    for dat_file in dat_files:
        with open(os.path.join(dat_folder, dat_file), 'r') as file:
            grid = []
            for line in file:
                grid.extend([int(num) for num in line.split()])
            sudoku_data.append(grid)
    return sudoku_data

# Paths 
sudoku_images_folder = "C:/Users/asaf0/OneDrive/maze_projecr/dataset"
dat_folder = "C:/Users/asaf0/OneDrive/maze_projecr/labels"

# Load the Sudoku grids (target output)
sudoku_labels = load_sudoku_dat_files(dat_folder)

# Extract the cells and corresponding labels (from Sudoku images)
image_files = [
    os.path.join(sudoku_images_folder, f)
    for f in os.listdir(sudoku_images_folder)
    if f.endswith(('.png', '.jpg', '.jpeg'))
]

# Initialize a list to store image cell data
image_cells = []
image_labels = []

for image_path, label in zip(image_files, sudoku_labels):
    cells = preprocess_and_extract_cells(image_path)
    image_cells.extend(cells)   
    image_labels.extend(label)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(image_cells, image_labels, test_size=0.2, random_state=42)

# Create PyTorch Datasets and DataLoaders
train_data = torch.utils.data.TensorDataset(torch.stack(X_train), torch.tensor(y_train))
test_data = torch.utils.data.TensorDataset(torch.stack(X_test), torch.tensor(y_test))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# Train the model
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop with accuracy, precision, and recall
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Collect predictions and labels for metrics
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculation per epoch
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, "
          f"Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}")

# Testing loop with metrics
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Collect predictions and labels for metrics
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculation for test
accuracy = 100 * correct / total
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

print(f"Test Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}")

# Save the trained model
torch.save(model.state_dict(), "sudoku_fully_connected_model.pth")

import random

# Recognize and print numbers from a random test image
if len(image_files) > 0:
    # Randomly choose an image from the test set
    random_index = random.randint(0, len(image_files) - 1)
    test_image_path = image_files[random_index]  # Select a random test image
    test_cells = preprocess_and_extract_cells(test_image_path)  # Extract cells from the test image

    # Recognize and print the numbers from the extracted cells
    print(f"Testing on image: {test_image_path}")
    recognize_and_print_numbers(test_cells, model)

    # Display the full image (entire Sudoku grid)
    image = cv2.imread(test_image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.title("Randomly Selected Sudoku Grid (Unseen by Model during Training)")
    plt.axis('off')
    plt.show()

    # Show the extracted cells 
    plt.figure(figsize=(10, 10))
    for i in range(9):
        for j in range(9):
            plt.subplot(9, 9, i * 9 + j + 1) 
            plt.imshow(test_cells[i * 9 + j].squeeze().numpy(), cmap="gray")
            plt.axis("off")
    plt.suptitle("Extracted Cells from the Randomly Selected Image")
    plt.show()

def is_valid(board, row, col, num):
    """Check if placing num at board[row][col] is valid."""
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
        # Check 3x3 subgrid
        if board[row // 3 * 3 + i // 3][col // 3 * 3 + i % 3] == num:
            return False
    return True

def solve_sudoku(board):
    """Solve the Sudoku using backtracking."""
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0: 
                for num in range(1, 10): 
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board):
                            return True
                        board[row][col] = 0 
                return False
    return True


def display_solved_sudoku(grid, original_image_path):
    """Display the solved Sudoku grid over the original image."""
    # Read the original image
    original_image = cv2.imread(original_image_path)
    original_image = cv2.resize(original_image, (450, 450))
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Overlay the solved grid numbers
    cell_size = 50
    for i in range(9):
        for j in range(9):
            num = grid[i][j]
            if num != 0:  # Draw only the solved numbers
                text = str(num)
                # Calculate the position for the number
                text_size = cv2.getTextSize(text, font, 1, 2)[0]
                text_x = j * cell_size + (cell_size - text_size[0]) // 2
                text_y = i * cell_size + (cell_size + text_size[1]) // 2
                cv2.putText(original_image, text, (text_x, text_y), font, 1, (0, 255, 0), 2)

    # Display the solved image
    plt.figure(figsize=(10, 10))
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title("Solved Sudoku Grid")
    plt.axis('off')
    plt.show()

# Recognize the Sudoku numbers
test_image_path = image_files[random.randint(0, len(image_files) - 1)]
test_cells = preprocess_and_extract_cells(test_image_path)
recognized_grid = []

for i in range(9):
    row = []
    for j in range(9):
        with torch.no_grad():
            cell = test_cells[i * 9 + j]
            output = model(cell)
            _, predicted = torch.max(output, 1)
            predicted_digit = predicted.item()
            row.append(predicted_digit if predicted_digit != 0 else 0)
    recognized_grid.append(row)

# Print the recognized grid
print("Recognized Sudoku Grid:")
for row in recognized_grid:
    print(row)

# Solve the Sudoku
solved_grid = [row[:] for row in recognized_grid]  # Make a copy of the recognized grid
if solve_sudoku(solved_grid):
    print("Solved Sudoku Grid:")
    for row in solved_grid:
        print(row)

    # Display the solved Sudoku grid over the original image
    display_solved_sudoku(solved_grid, test_image_path)
else:
    print("The Sudoku puzzle could not be solved.")




