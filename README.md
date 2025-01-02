# Thoughts-to-Text Converter

This project is an end-to-end pipeline for converting EEG signals into text using deep learning models. It comprises three main components:

1. **EEG Data Generator**: Simulates EEG signals for a vocabulary of words.
2. **EEG Processor**: Preprocesses signals, trains a model, and predicts text from signals.
3. **GUI Interface**: Provides an interactive interface for users to generate datasets, train the model, and test predictions.

---

## Features

- **Data Generation**: Simulates realistic EEG signals for a predefined vocabulary.
- **Signal Processing**: Includes bandpass filtering and normalization.
- **Deep Learning Model**: Uses an LSTM-based neural network for text prediction.
- **GUI**: A user-friendly interface built with Tkinter for ease of interaction.

---

## File Structure

### 1. `eeg_data_generator.py`

Generates synthetic EEG datasets with labeled word indices.

### 2. `eeg_processor.py`

Processes EEG signals and trains an LSTM model for classification.

### 3. `gui_interface.py`

Provides a graphical user interface for dataset generation, model training, and testing.

---

## Prerequisites

- Python 3.8+
- Required Libraries:
  - numpy
  - pandas
  - matplotlib
  - scipy
  - torch
  - tkinter

Install the dependencies using:

```bash
pip install numpy pandas matplotlib scipy torch
```

---

## Usage

### 1. Generate Dataset

Run the GUI and click on the **Generate Dataset** button to create a synthetic EEG dataset.

### 2. Train the Model

After generating the dataset, click **Train Model** to train the LSTM-based model on the data.

### 3. Test Prediction

Click **Test Prediction** to test the model’s ability to classify a random EEG signal from the dataset.

### 4. Run GUI

To launch the GUI, execute:

```bash
python gui_interface.py
```

---

## Example Output

- **Training Accuracy and Loss**: Visualized during the training process.
- **Prediction Results**: Displays predicted word, confidence level, and actual word in the GUI.

---

## Vocabulary

The model supports the following vocabulary:

- hello, world, thank, you, yes, no, goodbye, please, sorry, help
- maybe, love, friend, happy, sad

---

## Model Architecture

- **LSTM Layer**: Extracts temporal features from EEG sequences.
- **Dropout Layer**: Prevents overfitting.
- **Fully Connected Layer**: Maps features to vocabulary classes.

---

## How It Works

1. **Data Simulation**:

   - Generates synthetic EEG signals with unique patterns for each word.
   - Adds noise and variations to mimic real-world data.

2. **Signal Processing**:

   - Applies normalization and bandpass filtering.

3. **Model Training**:

   - Trains an LSTM-based neural network to classify EEG signals into words.

4. **Prediction**:

   - Predicts the word and confidence score for a given EEG signal.

---

## Future Enhancements

- Integration with real EEG hardware for live data input.
- Expanding the vocabulary.
- Experimenting with advanced neural network architectures like Transformers.

---
