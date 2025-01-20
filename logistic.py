import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
import torch

# Preprocess and extract Sudoku cells
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
            cell = cell.flatten() / 255.0 
            cells.append(cell)
    return cells

# Load Sudoku labels from .dat files
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

# Recognize and print numbers from cells
def recognize_and_print_numbers(cells, model):
    grid = []
    for cell in cells:
        cell_reshaped = cell.reshape(1, -1)
        predicted_digit = model.predict(cell_reshaped)[0]
        grid.append(predicted_digit if predicted_digit != 0 else 0)

sudoku_images_folder = "C:/Users/asaf0/OneDrive/sudoku_deepLearning/dataset"
dat_folder = "C:/Users/asaf0/OneDrive/sudoku_deepLearning/labels"

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

# Convert lists to numpy arrays
X = np.array(image_cells)
y = np.array(image_labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000, solver='lbfgs',random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate on test data
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
print("Testing Performance:")
print(f"Accuracy: {test_accuracy:.2f}")
print(f"Precision: {test_precision:.2f}")
print(f"Recall: {test_recall:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, zero_division=0))

# Save the trained model
import joblib
joblib.dump(model, "sudoku_logistic_model.pkl")

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
            plt.imshow(test_cells[i * 9 + j].reshape(28, 28), cmap="gray")
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
        cell = test_cells[i * 9 + j].reshape(1, -1)  
        predicted_digit = model.predict(cell)[0] 
        row.append(predicted_digit if predicted_digit != 0 else 0)
    recognized_grid.append(row)


# Solve the Sudoku
solved_grid = [row[:] for row in recognized_grid]
if solve_sudoku(solved_grid):

    # Display the solved Sudoku grid over the original image
    display_solved_sudoku(solved_grid, test_image_path)
else:
    print("The Sudoku puzzle could not be solved.")