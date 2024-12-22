from utils import preprocess_data, build_model, plot_training_history
import tensorflow as tf

# Paths
DATA_PATH = 'data/housing.csv'
MODEL_PATH = 'housing_price_model.h5'

# Step 1: Preprocess our data
x_train, x_test, y_train, y_test, scaler = preprocess_data(DATA_PATH)

# Step 2: Build the model - utils.py --> func "build model"
model = build_model(x_train.shape[1])

# Step 3: Train the model
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=2
)

# Step 4: Plot training history
plot_training_history(history)

# Step 5: Evaluate the model
loss, mae = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Step 6: Save the model
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")