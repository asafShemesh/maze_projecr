import random
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score


#logistic regression model
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(28 * 28, 10)  

    def forward(self, x):
        x = x.view(-1, 28 * 28)  
        x = self.linear(x)
        return x

#preprocess and extract Sudoku cells
def preprocess_and_extract_cells(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

#load Sudoku labels from .dat files
def load_sudoku_dat_files(dat_folder):
    dat_files = [f for f in os.listdir(dat_folder) if f.endswith('.dat')]
    sudoku_data = []
    for dat_file in dat_files:
        with open(os.path.join(dat_folder, dat_file), 'r') as file:
            grid = []
            for line in file:
                grid.extend([int(num) for num in line.split()])
            sudoku_data.append(grid)
    return sudoku_data

#recognize and print numbers from cells
def recognize_and_print_numbers(cells, model):
    grid = []
    for cell in cells:
        with torch.no_grad():
            output = model(cell)
            _, predicted = torch.max(output, 1)
            predicted_digit = predicted.item()
            grid.append(predicted_digit if predicted_digit != 0 else 0)
    
    for i in range(9):
        print(grid[i * 9:(i + 1) * 9])


sudoku_images_folder = "C:/Users/asaf0/OneDrive/maze_projecr/dataset"
dat_folder = "C:/Users/asaf0/OneDrive/maze_projecr/labels"

# Load Sudoku labels
sudoku_labels = load_sudoku_dat_files(dat_folder)

# Load images and preprocess them
image_files = [
    os.path.join(sudoku_images_folder, f)
    for f in os.listdir(sudoku_images_folder)
    if f.endswith(('.png', '.jpg', '.jpeg'))
]

image_cells = []
image_labels = []
for image_path, label in zip(image_files, sudoku_labels):
    cells = preprocess_and_extract_cells(image_path)
    image_cells.extend(cells)
    image_labels.extend(label)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(image_cells, image_labels, test_size=0.2, random_state=42)

# PyTorch Datasets and DataLoaders
train_data = torch.utils.data.TensorDataset(torch.stack(X_train), torch.tensor(y_train))
test_data = torch.utils.data.TensorDataset(torch.stack(X_test), torch.tensor(y_test))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# Model, loss function, and optimizer
model = LogisticRegressionModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

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
torch.save(model.state_dict(), "sudoku_logistic_model.pth")

# Test on a sample image
if len(image_files) > 0:
    test_image_path = image_files[0]
    test_cells = preprocess_and_extract_cells(test_image_path)
    recognize_and_print_numbers(test_cells, model)

    # Display the full Sudoku grid
    image = cv2.imread(test_image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.title("Full Sudoku Grid (Unseen by Model during Training)")
    plt.axis('off')
    plt.show()

    # Display extracted cells
    plt.figure(figsize=(10, 10))
    for i in range(9):
        for j in range(9):
            plt.subplot(9, 9, i * 9 + j + 1)
            plt.imshow(test_cells[i * 9 + j].squeeze().numpy(), cmap="gray")
            plt.axis("off")
    plt.suptitle("Extracted Cells from the First Image")
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
            if board[row][col] == 0:  # Empty cell
                for num in range(1, 10):  # Try digits 1-9
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board):
                            return True
                        board[row][col] = 0  # Undo move
                return False  # No valid number found
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
solved_grid = [row[:] for row in recognized_grid]
if solve_sudoku(solved_grid):
    print("Solved Sudoku Grid:")
    for row in solved_grid:
        print(row)

    # Display the solved Sudoku grid over the original image
    display_solved_sudoku(solved_grid, test_image_path)
else:
    print("The Sudoku puzzle could not be solved.")


