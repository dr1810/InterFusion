# InterFusion

This project implements a deep learning model for detecting fake news using multimodal inputs, including images, text headlines, and text descriptions. The model combines CNNs for image processing, BERT for text representation, and dense layers for categorical data to classify the news.

## Features
- **Multimodal Input**: Integrates image data, headline text, and descriptions.
- **BERT Integration**: Utilizes the Bangla-BERT model for processing Bengali text.
- **EfficientNet**: Leverages EfficientNetB0 for feature extraction from images.
- **Custom Fusion Layer**: Combines image, headline, and description features with learnable weights.
- **Regularization and Dropout**: Improves model generalization and reduces overfitting.

## Prerequisites
- Python 3.7 or higher
- TensorFlow 2.x
- Transformers library
- NLTK
- OpenCV
- pandas, numpy, matplotlib
- scikit-learn

## Setup Instructions
1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset and organize it into the following folders:
    ```
    ├── Train.csv
    ├── Validation.csv
    ├── Test.csv
    ├── Train/
    ├── Validation/
    ├── Test/
    ```

4. Download the Bangla-BERT pretrained model:
    ```bash
    pip install transformers
    ```

5. Download NLTK resources:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

## Files and Directories
- **`train.csv`, `validation.csv`, `test.csv`**: CSV files containing metadata (image ID, text, category labels).
- **`train/`, `validation/`, `test/`**: Directories containing the image data.
- **`main.py`**: Main script to preprocess data, train the model, and evaluate performance.

## Model Architecture
1. **Image Processing**: EfficientNetB0 is used to extract features from images.
2. **Text Processing**: Bangla-BERT processes headlines and descriptions.
3. **Categorical Data**: Dense layers process categorical inputs.
4. **Combination Layer**: Custom layer integrates all features into a unified representation.
5. **Output**: Single dense neuron with sigmoid activation for binary classification.

## Training
Run the training script:
```bash
python main.py


