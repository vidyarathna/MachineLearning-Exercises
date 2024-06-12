# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Reshape and normalize the pixel values
X_train = X_train.reshape((X_train.shape[0], -1)).astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], -1)).astype('float32') / 255
# Convert the labels to one-hot encoded format
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Create a neural network model
model = Sequential([
    Flatten(input_shape=(784,)),  # Flatten the input images (28x28) into a vector
    Dense(128, activation='relu'),  # First hidden layer with 128 units and ReLU activation
    Dense(64, activation='relu'),   # Second hidden layer with 64 units and ReLU activation
    Dense(10, activation='softmax')  # Output layer with 10 units for 10 classes and softmax activation
])

# Compile the model
model.compile(optimizer='adam',  # Adam optimizer
              loss='categorical_crossentropy',  # Categorical crossentropy loss for multiclass classification
              metrics=['accuracy'])  # Metric to monitor during training

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)  # Train for 10 epochs with a batch size of 32

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)  # Evaluate on the test data
print("Test Accuracy:", test_acc)

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')  # Plot training accuracy
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Plot validation accuracy
plt.xlabel('Epoch')  # Label for x-axis
plt.ylabel('Accuracy')  # Label for y-axis
plt.title('Training and Validation Accuracy')  # Title of the plot
plt.legend()  # Show legend
plt.show()  # Display the plot
