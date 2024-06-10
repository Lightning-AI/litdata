### Multi modal Model Image and Text

# Installation Guide:

sudo apt install libpoppler-cpp-dev
pip install python-poppler
sudo apt install libpoppler-cpp-dev poppler-utils
pip install fpdf python-poppler pdf2image names

# Attention:
Please note that the provided data and scripts are intended solely for trying out the code and demonstrating the workflow. The synthetic data used in this example is simplified and may not fully represent the complexity and variability of real-world customer emails and documents. For a proper evaluation and effective training of the model, it is crucial to use larger datasets with more diversity and noise. Real-world data often contains various imperfections, such as OCR errors, different document formats, and varied writing styles, which need to be accounted for to develop a robust and reliable model.

# Document Classification for Customer Emails

This project demonstrates a simple example of classifying customer email documents into three categories: cancellations, IBAN changes, and damage reports. The classification leverages both computer vision and natural language processing (NLP) techniques. Specifically, it combines the power of a BERT model for text analysis and a ResNet18 model for image recognition.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Generation](#data-generation)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Results](#results)
6. [Usage](#usage)
7. [Dependencies](#dependencies)
8. [License](#license)

## Introduction

In this example, we classify documents attached to customer emails and the email text itself into three classes:
- Cancellations
- IBAN Changes
- Damage Reports

We use a combined deep learning model where:
- A BERT model handles the NLP task of analyzing text.
- A ResNet18 model handles image recognition of the document.
- Both latent space representations are combined through a projection, followed by a classification head.

## Data Generation

The script includes a data generation module that simulates the creation of documents and their corresponding OCR text. This synthetic data is used to train and evaluate the model.

## Data Preparation

Data is processed and converted using `litdata`, which structures the data in a format suitable for training our deep learning model.

## Model Training

The combined model is trained on the prepared dataset. Training involves:
- Extracting text features using the BERT model.
- Extracting image features using the ResNet18 model.
- Combining these features in a projection layer.
- Classifying the combined features into the predefined classes.

## Evaluation

The model is evaluated using several metrics on the test dataset, including:
- Accuracy
- F1 Score
- Recall
- Precision

Additionally, the evaluation includes the generation of a confusion matrix and a detailed classification report, which are stored as CSV files. Predictions and logits are also saved for further analysis.

## Results

The evaluation results are stored in the following files:
- `test_confusion_matrix.csv`
- `test_classification_report.csv`
- `test_lables_predictions.csv`

## Usage

To use this code, follow these steps:

1. **Generate Data:**
    ```python
    python generate.py
    ```

2. **Prepare Data:**
    ```python
    python convert.py
    ```

3. **Train Model:**
    ```python
    python train.py
    ```

The scripts will handle data generation, preparation, training, and evaluation sequentially.

## Dependencies

Ensure you have the following dependencies installed:
- fpdf
- pdf2image
- names
- pyarrow
- pandas
- transformers
- litdata
- lightning
- joblib
- torchvision
- scikit-learn

You can install the necessary Python packages using:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the APACHE 2.0 License. See the [LICENSE](LICENSE) file for details.

---

For more detailed information on each step, please refer to the individual script files and the inline documentation provided within the code.