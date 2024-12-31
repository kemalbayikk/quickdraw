# QuickDraw Model Training

This repository contains a training pipeline for a Convolutional Neural Network (CNN) model designed to classify QuickDraw images. The project includes visualization, model creation, training, evaluation, and exporting the model to ONNX format.

## Features

- **Data Visualization**: Display sample images from each class to understand the dataset.
- **Model Architecture**: A CNN model with three convolutional blocks and fully connected layers.
- **Training Pipeline**: Trains the model with Adam optimizer and sparse categorical cross-entropy loss.
- **Evaluation**: Tests the model and provides accuracy and loss metrics.
- **Model Export**: Saves the model in HDF5 format and converts it to ONNX for compatibility.

## Requirements

- Python 3.10+
- TensorFlow
- scikit-learn
- Matplotlib
- NumPy
- tf2onnx
- onnx

## Usage

1. Clone the repository:
    ```bash
    git clone <repository_url>
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter Notebook to train the model and evaluate its performance.

## Files

- **training.ipynb**: Main notebook for training and exporting the model.
- **classes.txt**: Contains the class labels used in the dataset.
- **quickdraw_model_15_classes.h5**: Saved model in HDF5 format.
- **quickdraw_model_15_classes.onnx**: Exported ONNX model.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

