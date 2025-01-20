# Sudoku Deep Learning

This repository contains a deep learning project focused on recognizing, solving, and displaying Sudoku puzzles using computer vision and neural networks. The solution employs convolutional layers, fully connected layers, and additional regularization techniques like dropout to achieve robust results.

## Features
- Extract Sudoku grids from images using OpenCV.
- Recognize digits in each cell using a trained neural network.
- Solve the Sudoku puzzle using a backtracking algorithm.
- Display the solved puzzle overlayed on the original image.

## Project Structure
```
.
├── dataset/          # Folder containing Sudoku grid images
├── labels/           # Folder containing corresponding Sudoku solution labels (.dat files)
├── sudoku_deepLearning.py  # Main Python script for training and testing
├── requirements.txt  # Dependencies required for the project
├── README.md         # Project documentation
└── sudoku_logistic_model.pth  # Trained model (saved weights)
```

## Installation
### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/asafShemesh/sudoku_deepLearning.git
   cd sudoku_deepLearning
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```
   
## Usage
### Training the Model
To train the logistic regression or neural network model on your Sudoku dataset:
```bash
python sudoku_deepLearning.py
```

### Testing the Model
- A sample image from the test dataset will be processed, and the recognized Sudoku grid will be printed and solved.
- The solved grid will also be displayed overlayed on the original image.

### Input Format
- **Image Dataset**: Place Sudoku images in the `dataset/` folder.
- **Labels**: Ensure corresponding `.dat` files (Sudoku solutions) are in the `labels/` folder.

## Results
- **Recognition Accuracy**: Displays accuracy, precision, and recall metrics during training and testing.
- **Solved Sudoku Grid**: Outputs both the recognized and solved grids to the console and displays them visually.

## Example Output
### Recognized Sudoku Grid:
```
[5, 3, 0, 0, 7, 0, 0, 0, 0]
[6, 0, 0, 1, 9, 5, 0, 0, 0]
[0, 9, 8, 0, 0, 0, 0, 6, 0]
...
```

### Solved Sudoku Grid:
```
[5, 3, 4, 6, 7, 8, 9, 1, 2]
[6, 7, 2, 1, 9, 5, 3, 4, 8]
[1, 9, 8, 3, 4, 2, 5, 6, 7]
...
```

## Technologies Used
- **Python**: Programming language.
- **PyTorch**: For neural network modeling.
- **OpenCV**: For image preprocessing and grid extraction.
- **Matplotlib**: For visualizing results.
=======
# Sudoku Deep Learning

This repository contains a deep learning project focused on recognizing, solving, and displaying Sudoku puzzles using computer vision and neural networks. The solution employs convolutional layers, fully connected layers, and additional regularization techniques like dropout to achieve robust results.

## Features
- Extract Sudoku grids from images using OpenCV.
- Recognize digits in each cell using a trained neural network.
- Solve the Sudoku puzzle using a backtracking algorithm.
- Display the solved puzzle overlayed on the original image.

## Project Structure
```
.
├── dataset/          # Folder containing Sudoku grid images
├── labels/           # Folder containing corresponding Sudoku solution labels (.dat files)
├── sudoku_deepLearning.py  # Main Python script for training and testing
├── requirements.txt  # Dependencies required for the project
├── README.md         # Project documentation
└── sudoku_logistic_model.pth  # Trained model (saved weights)
```

## Installation
### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/asafShemesh/sudoku_deepLearning.git
   cd sudoku_deepLearning
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```
   
## Usage
### Training the Model
To train the logistic regression or neural network model on your Sudoku dataset:
```bash
python sudoku_deepLearning.py
```

### Testing the Model
- A sample image from the test dataset will be processed, and the recognized Sudoku grid will be printed and solved.
- The solved grid will also be displayed overlayed on the original image.

### Input Format
- **Image Dataset**: Place Sudoku images in the `dataset/` folder.
- **Labels**: Ensure corresponding `.dat` files (Sudoku solutions) are in the `labels/` folder.

## Results
- **Recognition Accuracy**: Displays accuracy, precision, and recall metrics during training and testing.
- **Solved Sudoku Grid**: Outputs both the recognized and solved grids to the console and displays them visually.

## Example Output
### Recognized Sudoku Grid:
```
[5, 3, 0, 0, 7, 0, 0, 0, 0]
[6, 0, 0, 1, 9, 5, 0, 0, 0]
[0, 9, 8, 0, 0, 0, 0, 6, 0]
...
```

### Solved Sudoku Grid:
```
[5, 3, 4, 6, 7, 8, 9, 1, 2]
[6, 7, 2, 1, 9, 5, 3, 4, 8]
[1, 9, 8, 3, 4, 2, 5, 6, 7]
...
```

## Technologies Used
- **Python**: Programming language.
- **PyTorch**: For neural network modeling.
- **OpenCV**: For image preprocessing and grid extraction.
- **Matplotlib**: For visualizing results.
>>>>>>> f7552226b9ea094e6adbffd59294c97837d83361
- **Scikit-learn**: For train-test splitting and evaluation metrics.