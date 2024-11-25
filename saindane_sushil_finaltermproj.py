import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Load the dataset
print("Loading the dataset...")
file_path = r"F:\SEM 3\CS634_DataMining\Final_Project\PhiUSIIL_Phishing_URL_Dataset.csv"
raw_data = pd.read_csv(file_path, nrows=50000)  # Load 50,000 rows
print("Dataset loaded. Shape:", raw_data.shape)

# Function to preprocess data
def preprocess_data(df):
    # Handle high-cardinality features
    high_cardinality_features = ['FILENAME', 'URL', 'Domain']
    hashed_features = []
    for feature in high_cardinality_features:
        hasher = FeatureHasher(n_features=100, input_type='string')
        hashed = hasher.transform([[str(val)] for val in df[feature]])
        hashed_features.append(hashed.toarray())
    
    # Handle low-cardinality features
    low_cardinality_features = ['TLD', 'Title']
    encoded_features = pd.get_dummies(df[low_cardinality_features], prefix=low_cardinality_features)
    
    # Select numeric features
    numeric_features = df.select_dtypes(include=[np.number]).drop('label', axis=1, errors='ignore')
    
    # Combine all features
    X = np.hstack([np.hstack(hashed_features), encoded_features, numeric_features])
    y = df['label'].values
    
    return X, y

# Preprocess data
print("\nProcessing data...")
X, y = preprocess_data(raw_data)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data preprocessing complete.")
print("Training set shape:", X_train_scaled.shape)
print("Testing set shape:", X_test_scaled.shape)

# Binarize features for Bernoulli Naive Bayes
binarizer = Binarizer()
X_train_binarized = binarizer.fit_transform(X_train)
X_test_binarized = binarizer.transform(X_test)

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    tss = tpr - fpr
    hss = 2 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) if ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)) > 0 else 0
    
    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1,
        'TPR': tpr, 'TNR': tnr, 'FPR': fpr, 'FNR': fnr,
        'TSS': tss, 'HSS': hss
    }

def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    print(f"\nTraining and evaluating {model_name}...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_index, val_index) in enumerate(tqdm(kf.split(X_train), total=10, desc=f"{model_name} Folds")):
        X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
        
        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_fold_val)
        
        fold_metrics.append(calculate_metrics(y_fold_val, y_pred))
    
    # Calculate average metrics
    avg_metrics = {metric: np.mean([fold[metric] for fold in fold_metrics]) for metric in fold_metrics[0]}
    
    # Evaluate on test set
    y_pred_test = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    return fold_metrics, avg_metrics, test_metrics

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_fold_metrics, rf_avg_metrics, rf_test_metrics = train_and_evaluate(rf_model, X_train_scaled, X_test_scaled, y_train, y_test, 'Random Forest')

# Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_fold_metrics, dt_avg_metrics, dt_test_metrics = train_and_evaluate(dt_model, X_train_scaled, X_test_scaled, y_train, y_test, 'Decision Tree')

# Train LSTM
def train_and_evaluate_lstm(X_train, X_test, y_train, y_test):
    print("\nTraining and evaluating LSTM...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_index, val_index) in enumerate(tqdm(kf.split(X_train), total=10, desc="LSTM Folds")):
        X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
        
        X_fold_train = X_fold_train.reshape((X_fold_train.shape[0], 1, X_fold_train.shape[1]))
        X_fold_val = X_fold_val.reshape((X_fold_val.shape[0], 1, X_fold_val.shape[1]))
        
        lstm_model = Sequential([
            Input(shape=(1, X_train.shape[1])),
            LSTM(64),
            Dense(1, activation='sigmoid')
        ])
        lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        lstm_model.fit(X_fold_train, y_fold_train, epochs=5, batch_size=32, verbose=0)
        y_pred = (lstm_model.predict(X_fold_val) > 0.5).astype(int).flatten()
        
        fold_metrics.append(calculate_metrics(y_fold_val, y_pred))
    
    # Calculate average metrics
    avg_metrics = {metric: np.mean([fold[metric] for fold in fold_metrics]) for metric in fold_metrics[0]}
    
    # Evaluate on test set
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    y_pred_test = (lstm_model.predict(X_test_reshaped) > 0.5).astype(int).flatten()
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    return fold_metrics, avg_metrics, test_metrics

lstm_fold_metrics, lstm_avg_metrics, lstm_test_metrics = train_and_evaluate_lstm(X_train_scaled, X_test_scaled, y_train, y_test)

# Train Bernoulli Naive Bayes
bnb_model = BernoulliNB()
bnb_fold_metrics, bnb_avg_metrics, bnb_test_metrics = train_and_evaluate(bnb_model, X_train_binarized, X_test_binarized, y_train, y_test, 'Bernoulli Naive Bayes')

# Save results
np.save('rf_results.npy', {'fold_metrics': rf_fold_metrics, 'avg_metrics': rf_avg_metrics, 'test_metrics': rf_test_metrics})
np.save('dt_results.npy', {'fold_metrics': dt_fold_metrics, 'avg_metrics': dt_avg_metrics, 'test_metrics': dt_test_metrics})
np.save('lstm_results.npy', {'fold_metrics': lstm_fold_metrics, 'avg_metrics': lstm_avg_metrics, 'test_metrics': lstm_test_metrics})
np.save('bnb_results.npy', {'fold_metrics': bnb_fold_metrics, 'avg_metrics': bnb_avg_metrics, 'test_metrics': bnb_test_metrics})
print("Results saved.")

# Function to create tabular metrics
def create_metrics_table(fold_metrics, avg_metrics, test_metrics, model_name):
    table_data = []
    headers = ["Fold", "Accuracy", "Precision", "Recall", "F1-score", "TPR", "TNR", "FPR", "FNR", "TSS", "HSS"]
    
    for i, fold in enumerate(fold_metrics, 1):
        table_data.append([
            f"Fold {i}",
            f"{fold['Accuracy']:.4f}",
            f"{fold['Precision']:.4f}",
            f"{fold['Recall']:.4f}",
            f"{fold['F1-score']:.4f}",
            f"{fold['TPR']:.4f}",
            f"{fold['TNR']:.4f}",
            f"{fold['FPR']:.4f}",
            f"{fold['FNR']:.4f}",
            f"{fold['TSS']:.4f}",
            f"{fold['HSS']:.4f}"
        ])
    
    table_data.append([
        "Average",
        f"{avg_metrics['Accuracy']:.4f}",
        f"{avg_metrics['Precision']:.4f}",
        f"{avg_metrics['Recall']:.4f}",
        f"{avg_metrics['F1-score']:.4f}",
        f"{avg_metrics['TPR']:.4f}",
        f"{avg_metrics['TNR']:.4f}",
        f"{avg_metrics['FPR']:.4f}",
        f"{avg_metrics['FNR']:.4f}",
        f"{avg_metrics['TSS']:.4f}",
        f"{avg_metrics['HSS']:.4f}"
    ])
    
    table_data.append([
        "Test",
        f"{test_metrics['Accuracy']:.4f}",
        f"{test_metrics['Precision']:.4f}",
        f"{test_metrics['Recall']:.4f}",
        f"{test_metrics['F1-score']:.4f}",
        f"{test_metrics['TPR']:.4f}",
        f"{test_metrics['TNR']:.4f}",
        f"{test_metrics['FPR']:.4f}",
        f"{test_metrics['FNR']:.4f}",
        f"{test_metrics['TSS']:.4f}",
        f"{test_metrics['HSS']:.4f}"
    ])
    
    print(f"\n{model_name} Metrics:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

# Function to plot confusion matrix
def plot_confusion_matrix(test_metrics, model_name):
    cm = np.array([[test_metrics['TN'], test_metrics['FP']],
                   [test_metrics['FN'], test_metrics['TP']]])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Create tables and confusion matrices for all models
create_metrics_table(rf_fold_metrics, rf_avg_metrics, rf_test_metrics, "Random Forest")
plot_confusion_matrix(rf_test_metrics, "Random Forest")

create_metrics_table(dt_fold_metrics, dt_avg_metrics, dt_test_metrics, "Decision Tree")
plot_confusion_matrix(dt_test_metrics, "Decision Tree")

create_metrics_table(lstm_fold_metrics, lstm_avg_metrics, lstm_test_metrics, "LSTM")
plot_confusion_matrix(lstm_test_metrics, "LSTM")

create_metrics_table(bnb_fold_metrics, bnb_avg_metrics, bnb_test_metrics, "Bernoulli Naive Bayes")
plot_confusion_matrix(bnb_test_metrics, "Bernoulli Naive Bayes")

# Comparison table
comparison_data = [
    ["Model", "Accuracy", "Precision", "Recall", "F1-score", "TPR", "TNR", "FPR", "FNR", "TSS", "HSS"],
    ["Random Forest", rf_test_metrics["Accuracy"], rf_test_metrics["Precision"], 
     rf_test_metrics["Recall"], rf_test_metrics["F1-score"], 
     rf_test_metrics["TPR"], rf_test_metrics["TNR"], 
     rf_test_metrics["FPR"], rf_test_metrics["FNR"], 
     rf_test_metrics["TSS"], rf_test_metrics["HSS"]],
    ["Decision Tree", dt_test_metrics["Accuracy"], dt_test_metrics["Precision"], 
     dt_test_metrics["Recall"], dt_test_metrics["F1-score"], 
     dt_test_metrics["TPR"], dt_test_metrics["TNR"], 
     dt_test_metrics["FPR"], dt_test_metrics["FNR"], 
     dt_test_metrics["TSS"], dt_testmetrics["HSS"]],
    ["LSTM", lstm_testmetrics["Accuracy"], lstm_testmetrics["Precision"], 
     lstm_testmetrics["Recall"], lstm_testmetrics["F1-score"], 
     lstm_testmetrics["TPR"], lstm_testmetrics["TNR"], 
     lstm_testmetrics["FPR"], lstm_testmetrics["FNR"], 
     lstm_testmetrics["TSS"], lstm_testmetrics["HSS"]],
    ["Bernoulli Naive Bayes", bnb_testmetrics["Accuracy"], bnb_testmetrics["Precision"],
     bnb_testmetrics["Recall"], bnb_testmetrics["F1-score"],
     bnb_testmetrics["TPR"], bnb_testmetrics["TNR"],
     bnb_testmetrics["FPR"], bnb_testmetrics["FNR"],
     bnb_testmetrics["TSS"], bnb_testmetrics["HSS"]]
]

print("\nComparison of Models:")
print(tabulate(comparison_data, headers="firstrow", tablefmt="grid"))

