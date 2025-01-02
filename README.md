# EEG Signal Classification

This repository contains an implementation of a neural network model for EEG signal classification. The system is designed to preprocess raw EEG data, train a model to recognize patterns, and predict the class of new EEG signals. 

## Features
- **Signal Preprocessing**: Includes normalization, filtering, and preparation of raw EEG data.
- **Customizable Neural Network**: Allows for flexible architecture design based on the number of target classes.
- **Training and Validation**: Tracks loss and accuracy over multiple epochs.
- **Inference with Confidence**: Predicts class labels and provides confidence scores.
- **Model Persistence**: Save and load trained models for future use.
- **Performance Visualization**: Plots training and validation metrics over epochs.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/eeg-signal-classification.git
   cd eeg-signal-classification
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Preprocess Signals
Raw EEG signals are preprocessed to ensure they are ready for training and prediction.
```python
processed_signal = eeg_processor.preprocess_signal(raw_signal)
```

### 2. Train the Model
Train the model on your dataset with specified parameters.
```python
history = eeg_processor.train(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
```

### 3. Make Predictions
Use the trained model to predict the class of new EEG signals.
```python
predicted_class, confidence = eeg_processor.predict(new_signal, return_confidence=True)
print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")
```

### 4. Save and Load Model
Save the trained model:
```python
eeg_processor.save_model('model.pth')
```
Load a pre-trained model:
```python
eeg_processor.load_model('model.pth')
```

### 5. Plot Training History
Visualize training and validation metrics:
```python
eeg_processor.plot_training_history(history)
```

## Example Workflow

```python
from eeg_signal_classification import EEGProcessor

# Initialize processor
processor = EEGProcessor(device='cuda')

# Load data
X_train, y_train = load_training_data()

# Train the model
history = processor.train(X_train, y_train, epochs=50, batch_size=32)

# Save the model
processor.save_model('eeg_model.pth')

# Predict new signal
signal = load_new_signal()
prediction, confidence = processor.predict(signal, return_confidence=True)
print(f"Class: {prediction}, Confidence: {confidence:.2f}")
```

## Requirements
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

Install dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Contributing
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgements
- The project is inspired by the challenges in EEG signal processing and classification.
- Thanks to the open-source community for tools and frameworks that made this project possible.

---
Developed with ❤️ by Aditya and contributors.
