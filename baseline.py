import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

def preprocess_and_extract_cells(image_path):
    """Preprocess the Sudoku image, extract the grid, and split it into cells."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    grid = binary[y:y + h, x:x + w]

    grid = cv2.resize(grid, (450, 450))  # Resize to standard size
    cells = []
    cell_size = 50  # Each cell is 50x50 pixels
    for i in range(9):
        for j in range(9):
            cell = grid[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            cell = cv2.resize(cell, (28, 28))  # Resize to 28x28 for digit recognition
            cell = torch.tensor(cell, dtype=torch.float32).unsqueeze(0) / 255.0
            cells.append(cell)

    return cells

def recognize_and_print_numbers_random(cells):
    """Generate random numbers for Sudoku cells and print the grid."""
    grid = []
    for _ in cells:
        random_digit = random.randint(0, 9)  # Generate random digit between 0 and 9
        grid.append(random_digit)

    # Print the 9x9 grid
    for i in range(9):
        print(grid[i * 9:(i + 1) * 9])

# Paths to the dataset
sudoku_images_folder = "C:/Users/asaf0/OneDrive/maze_projecr/dataset"  # Path to Sudoku images

# Extract the cells from the first image for demonstration purposes
image_files = [
    os.path.join(sudoku_images_folder, f)
    for f in os.listdir(sudoku_images_folder)
    if f.endswith(('.png', '.jpg', '.jpeg'))
]

if len(image_files) > 0:
    test_image_path = image_files[0]  # Use the first image for testing
    test_cells = preprocess_and_extract_cells(test_image_path)  # Extract cells from the test image

    # Generate and print random numbers for the cells
    recognize_and_print_numbers_random(test_cells)

    # Display the full image (entire Sudoku grid)
    image = cv2.imread(test_image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct display
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.title("Full Sudoku Grid (Unseen by Model during Training)")
    plt.axis('off')  # Hide axes
    plt.show()

    # Show the extracted cells arranged in a 9x9 grid
    plt.figure(figsize=(10, 10))
    for i in range(9):
        for j in range(9):
            plt.subplot(9, 9, i * 9 + j + 1)  # Place each cell in a 9x9 grid
            plt.imshow(test_cells[i * 9 + j].squeeze().numpy(), cmap="gray")
            plt.axis("off")
    plt.suptitle("Extracted Cells from the First Image")
    plt.show()

    # Calculate and display accuracy (dummy example for random model)
    true_labels = [random.randint(0, 9) for _ in range(81)]  # Replace with actual labels if available
    predictions = [random.randint(0, 9) for _ in range(81)]
    correct = sum([1 for true, pred in zip(true_labels, predictions) if true == pred])
    accuracy = correct / len(true_labels) * 100
    print(f"Accuracy of random predictions: {accuracy:.2f}%")
