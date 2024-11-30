# Import Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


def preprocess_data(file_path):
    # Import the dataset and assign it to "data"
    data = pd.read_csv(file_path)

    # Converts categorical variables to binary format using one hot encoding
    data = pd.get_dummies(data, drop_first=True)

    # Separate features (X) and target (y)
    X = data.drop(columns=['price'])
    y = data['price']

    # Split into training and test sets
    x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize numerical features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(X_test)

    return x_train, X_test, y_train, y_test, scaler


def build_model(input_shape):
    # Create a sequential neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    # ^^^ Output layer for regression
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
