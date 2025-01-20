# Reimport necessary libraries after reset
import os
import numpy as np
import matplotlib.pyplot as plt
import zipfile

# Redefine necessary functions

# Function to check if a number can be placed in the Sudoku grid
def is_valid_move(grid, row, col, num):
    size = len(grid)
    base = int(size ** 0.5)
    if num in grid[row, :] or num in grid[:, col]:
        return False
    start_row, start_col = (row // base) * base, (col // base) * base
    subgrid = grid[start_row:start_row + base, start_col:start_col + base]
    if num in subgrid:
        return False
    return True

# Function to solve a Sudoku grid using backtracking
def solve_sudoku(grid):
    for row in range(len(grid)):
        for col in range(len(grid)):
            if grid[row, col] == 0:
                for num in range(1, len(grid) + 1):
                    if is_valid_move(grid, row, col, num):
                        grid[row, col] = num
                        if solve_sudoku(grid):
                            return True
                        grid[row, col] = 0
                return False
    return True

# Function to create a valid Sudoku puzzle
def create_valid_sudoku(size=9, num_clues=30):
    base = int(size ** 0.5)
    grid = np.zeros((size, size), dtype=int)
    for i in range(0, size, base):
        nums = np.random.permutation(range(1, size + 1))
        for j in range(base):
            for k in range(base):
                grid[i + j, i + k] = nums[j * base + k]
    solve_sudoku(grid)
    puzzle = grid.copy()
    total_cells = size * size
    indices = np.random.choice(total_cells, size=total_cells - num_clues, replace=False)
    for idx in indices:
        row, col = divmod(idx, size)
        puzzle[row, col] = 0
    return puzzle, grid

# Function to draw Sudoku
def draw_sudoku_proper(sudoku, filename):
    size = sudoku.shape[0]
    base = int(size**0.5)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    for x in range(size + 1):
        linewidth = 2 if x % base == 0 else 0.5
        ax.axhline(x, color="black", linewidth=linewidth)
        ax.axvline(x, color="black", linewidth=linewidth)
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    for (row, col), value in np.ndenumerate(sudoku):
        if value != 0:
            ax.text(
                col + 0.5, size - row - 0.5, str(value),
                va="center", ha="center", fontsize=16, color="black"
            )
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

# Create a dataset with 50 valid Sudoku puzzles and unique file names
output_dir = "./sudoku_dataset_50_valid_unique"
os.makedirs(output_dir, exist_ok=True)

image_files = []
dat_files = []

for i in range(1000):
    # Create a valid Sudoku puzzle and its solution
    puzzle, solution = create_valid_sudoku(size=9, num_clues=30)

    # Save the puzzle matrix as a .dat file with a unique name
    dat_filename = os.path.join(output_dir, f"z1sudoku_{i + 1}.dat")
    np.savetxt(dat_filename, puzzle, fmt="%d")
    dat_files.append(dat_filename)

    # Save the puzzle as an image with a unique name
    image_filename = os.path.join(output_dir, f"z1sudoku_{i + 1}.png")
    draw_sudoku_proper(puzzle, image_filename)
    image_files.append(image_filename)

# Create a zip file with all generated files
zip_filename = "./sudoku_dataset_50_valid_unique3.zip"
with zipfile.ZipFile(zip_filename, 'w') as zipf:
    for file in image_files + dat_files:
        zipf.write(file, os.path.basename(file))

# Clean up temporary files
for file in image_files + dat_files:
    os.remove(file)

