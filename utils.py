# Import Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
