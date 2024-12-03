# CS634 Data Mining Final Project: Phishing URL Detection

## Project Overview

This project implements and compares four classification algorithms for detecting phishing URLs using the PhiUSIIL Phishing URL Dataset. The algorithms are evaluated using 10-fold cross-validation and various performance metrics.

## Algorithms Implemented

1. Random Forest
2. Decision Tree
3. LSTM (Long Short-Term Memory)
4. Bernoulli Naive Bayes

## Features

- Implements four different classification algorithms
- Utilizes the PhiUSIIL Phishing URL Dataset
- Performs data preprocessing including feature hashing and one-hot encoding
- Implements 10-fold cross-validation for model evaluation
- Calculates and compares various performance metrics (Accuracy, Precision, Recall, F1-score, TPR, TNR, FPR, FNR, TSS, HSS)
- Generates confusion matrices for visual performance analysis

## Requirements

- Python 3.x
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - tensorflow
  - matplotlib
  - seaborn
  - tabulate
  - tqdm

## Installation

Install the required packages using pip: 
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn tabulate tqdm

## Usage

1. Ensure the dataset file `PhiUSIIL_Phishing_URL_Dataset.csv` is in the same directory as the script.
2. Run the Python script:
   python saindane_sushil_finaltermproj.py
3. The program will execute the following steps:
    - Load and preprocess the dataset (50,000 rows)
    - Split the data into training and testing sets
    - Scale the features using StandardScaler
    - Binarize features for Bernoulli Naive Bayes
4. For each model (Random Forest, Decision Tree, LSTM, Bernoulli Naive Bayes):
    - Perform 10-fold cross-validation
    - Train the model on each fold
    - Calculate performance metrics for each fold
    - Evaluate the model on the test set
5. The program will display:
    - Progress bars for data processing and model training
    - Detailed metrics tables for each model, including per-fold results, averages, and test set performance
    - Confusion matrices for visual analysis of each model's performance
6. Results will be saved as .npy files in the same directory:
    - `rf_results.npy` for Random Forest
    - `dt_results.npy` for Decision Tree
    - `lstm_results.npy` for LSTM
    - `bnb_results.npy` for Bernoulli Naive Bayes
7. A final comparison table of all models' performance on the test set will be displayed.

Note: The entire process may take some time to complete, especially the LSTM training. Ensure you have sufficient computational resources available.

Or open and run the Jupyter notebook `saindane_sushil_finaltermproj.ipynb`.

## Project Structure

- `saindane_sushil_finaltermproj.py`: Main Python script
- `saindane_sushil_finaltermproj.ipynb`: Jupyter notebook version
- `PhiUSIIL_Phishing_URL_Dataset.csv`: Dataset file
## Results

The project outputs:
- Detailed metrics for each algorithm (per fold, average, and test set)
- Confusion matrices for each algorithm
- Comparison table of all models' performance on the test set

## Author

Sushil Saindane

## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

Sushil Saindane - sbs8@njit.edu

LinkedIn - https://www.linkedin.com/in/sushil-saindane-12520a1a4/

Project Link: https://github.com/sushilsaindane/saindane_sushil_finaltermproj



