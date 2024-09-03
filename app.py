import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# List of company tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']

# Load the dataset and models
def load_data(ticker):
    # Fetch historical stock data
    df = yf.download(ticker, start='2015-01-01', end='2023-01-01')
    data = df['Close'].values.reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare the data for time series prediction
    def create_dataset(dataset, look_back=60):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            X.append(dataset[i:(i + look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    
    look_back = 60
    X, y = create_dataset(scaled_data, look_back)
    
    # Reshape data for CNN, LSTM, and GRU
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split data into training and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_test, y_test, scaler

def evaluate_classification_metrics(y_true, y_pred):
    # Binary classification thresholding
    y_true_class = np.where(y_true > np.mean(y_true), 1, 0)
    y_pred_class = np.where(y_pred > np.mean(y_pred), 1, 0)
    
    accuracy = accuracy_score(y_true_class, y_pred_class)
    precision = precision_score(y_true_class, y_pred_class)
    recall = recall_score(y_true_class, y_pred_class)
    f1 = f1_score(y_true_class, y_pred_class)
    
    return accuracy, precision, recall, f1

def plot_confusion_matrix(y_true, y_pred, title):
    y_true_class = np.where(y_true > np.mean(y_true), 1, 0)
    y_pred_class = np.where(y_pred > np.mean(y_pred), 1, 0)
    
    cm = confusion_matrix(y_true_class, y_pred_class)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    st.pyplot(fig)

def plot_metrics_graphs(metrics_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics_df['Model'], metrics_df['Accuracy'], marker='o', label='Accuracy')
    ax.plot(metrics_df['Model'], metrics_df['Precision'], marker='o', label='Precision')
    ax.plot(metrics_df['Model'], metrics_df['Recall'], marker='o', label='Recall')
    ax.plot(metrics_df['Model'], metrics_df['F1-Score'], marker='o', label='F1-Score')

    ax.set_title('Comparative Analysis of Metrics')
    ax.set_xlabel('Model')
    ax.set_ylabel('Metric Value')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Line graph for accuracy only
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics_df['Model'], metrics_df['Accuracy'], marker='o', label='Accuracy')
    ax.set_title('Comparative Analysis of Accuracy')
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def main():
    st.set_page_config(page_title='Stock Trading Analysis', layout='wide')
    
    # Load models
    cnn_model = load_model('cnn_model.keras')
    lstm_model = load_model('lstm_model.keras')
    gru_model = load_model('gru_model.keras')

    # Dropdown for company selection
    ticker = st.sidebar.selectbox('Select a Company', tickers)

    # Load data for the selected company
    X_test, y_test, scaler = load_data(ticker)

    # Make predictions
    cnn_predictions = scaler.inverse_transform(cnn_model.predict(X_test))
    lstm_predictions = scaler.inverse_transform(lstm_model.predict(X_test))
    gru_predictions = scaler.inverse_transform(gru_model.predict(X_test))

    # Evaluate models
    cnn_metrics = evaluate_classification_metrics(y_test, cnn_predictions)
    lstm_metrics = evaluate_classification_metrics(y_test, lstm_predictions)
    gru_metrics = evaluate_classification_metrics(y_test, gru_predictions)
    
    metrics_data = {
        'Model': ['CNN', 'LSTM', 'GRU'],
        'Accuracy': [cnn_metrics[0], lstm_metrics[0], gru_metrics[0]],
        'Precision': [cnn_metrics[1], lstm_metrics[1], gru_metrics[1]],
        'Recall': [cnn_metrics[2], lstm_metrics[2], gru_metrics[2]],
        'F1-Score': [cnn_metrics[3], lstm_metrics[3], gru_metrics[3]]
    }

    metrics_df = pd.DataFrame(metrics_data)
    
    # Best model
    best_model_index = metrics_df['Accuracy'].idxmax()
    best_model = metrics_df.iloc[best_model_index]['Model']

    # Set home page with background image
    st.markdown(
        """
        <style>
        .reportview-container {
            background: url("background_image.jpg");
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Dropdown for navigation
    option = st.sidebar.selectbox(
        'Select a Section',
        ['Confusion Matrix', 'Result Table', 'Result Graph']
    )

    if option == 'Confusion Matrix':
        st.header('Confusion Matrix')
        
        st.subheader('CNN Model')
        plot_confusion_matrix(y_test, cnn_predictions, title=f"CNN Confusion Matrix ({ticker})")
        
        st.subheader('LSTM Model')
        plot_confusion_matrix(y_test, lstm_predictions, title=f"LSTM Confusion Matrix ({ticker})")
        
        st.subheader('GRU Model')
        plot_confusion_matrix(y_test, gru_predictions, title=f"GRU Confusion Matrix ({ticker})")

    elif option == 'Result Table':
        st.header('Result Table')
        st.subheader('Comparative Table for All Metrics')
        st.write(metrics_df)

        st.subheader('Comparative Table for Accuracy Only')
        accuracy_df = metrics_df[['Model', 'Accuracy']]
        st.write(accuracy_df)
        
        st.subheader('Best Model')
        st.write(f"The best model based on accuracy is: {best_model}")

    elif option == 'Result Graph':
        st.header('Result Graph')
        
        st.subheader('Line Graph for All Metrics')
        plot_metrics_graphs(metrics_df)
        
        st.subheader('Best Model')
        st.write(f"The best model based on accuracy is: {best_model}")

if __name__ == "__main__":
    main()
