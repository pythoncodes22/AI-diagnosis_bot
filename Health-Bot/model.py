import numpy as np
import tensorflow as tf
import asyncio
input_size = 3 
hidden_layers = 8  # Number of hidden layers
hidden_units = 512  # Number of neurons in each hidden layer
output_size = 3  # Number of possible moves in Monopoly

diagnosis_descriptions = {
    1: "Diagnosis type 1: Go to the doctor.",
    2: "Diagnosis type 2: Don't go to the doctor, you should be fine.",
    3: "Diagnosis type 3: Go to the hospital.",
}

# Define the training data
training_data = [
    {"test": [1, 1, 2], "diagnosis": 2},
    {"test": [3, 2, 6], "diagnosis": 1},
    {"test": [1, 1, 4], "diagnosis": 1},
    {"test": [2, 1, 1], "diagnosis": 2},
    {"test": [1, 10, 10], "diagnosis": 3},
    {"test": [3, 1, 1], "diagnosis": 2},  # Example 1: Game state [5, 10, 9], Diagnosis: 3

    # Add more training data examples as needed
]
# Define rewards associated with each training example
rewards = [1, 1, 1]  # Example rewards, adjust according to your application

# Data augmentation function
def augment_data(data, num_augmentations):
    augmented_data = []
    for example in data:
        for _ in range(num_augmentations):
            augmented_example = {
                "test": np.clip(np.array(example["test"]) + np.random.normal(0, 0.1, size=len(example["test"])), 0, 10),
                "diagnosis": example["diagnosis"]
            }
            augmented_data.append(augmented_example)
    return augmented_data

# Augment the training data
augmented_training_data = augment_data(training_data, num_augmentations=10)
augmented_rewards = rewards * 10
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, hidden_units, output_size):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.output_size = output_size

        model_layers = [
            tf.keras.layers.Dense(hidden_units, input_shape=(input_size,), activation='relu'),
            tf.keras.layers.BatchNormalization(),  # Batch normalization layer
            tf.keras.layers.Dropout(0.5)  # Dropout layer with 50% dropout rate
        ]

        for _ in range(hidden_layers - 1):
            model_layers.append(tf.keras.layers.Dense(hidden_units, activation='relu'))
            model_layers.append(tf.keras.layers.BatchNormalization())  # Batch normalization layer
            model_layers.append(tf.keras.layers.Dropout(0.5))  # Dropout layer with 50% dropout rate

        model_layers.append(tf.keras.layers.Dense(output_size, activation='softmax'))

        self.model = tf.keras.Sequential(model_layers)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def predict(self, input_vector):
        return self.model.predict(input_vector)

    def train(self, training_data, epochs, rewards):
        input_data = np.array([example['test'] for example in training_data])
        target_data = np.array(
            [example['diagnosis'] - 1 for example in training_data])  # Adjust move indices to start from 0

        self.model.fit(input_data, target_data, epochs=epochs, sample_weight=np.array(rewards), verbose=0)

