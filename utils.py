"""
Utility functions and classes for Jupyter Notebooks lessons using TensorFlow.
"""

from typing import List, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging
from logging import INFO, ERROR
from flwr.common.logger import console_handler, log

from flwr.common.logger import console_handler
from flwr.common import Metrics

class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == INFO

console_handler = logging.StreamHandler()
console_handler.setLevel(INFO)
console_handler.addFilter(InfoFilter())

# To filter logging coming from TensorFlow
tf.get_logger().setLevel(ERROR)

def get_datasets(client_conf):
    """
    Splits the MNIST dataset into subsets based on the given client configuration.

    Args:
        client_conf (list): Configuration for each client, determining which digits to include.

    Returns:
        Tuple: A tuple containing the training sets for each client and the test set.
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0  # Normalize
    x_test = x_test / 255.0
    
    N_CLIENTS = len(client_conf)
    total_length = len(x_train)
    
    print(f"Total samples in MNIST: {total_length}")
    
    np.random.seed(42)
    
    indices = np.arange(total_length)
    np.random.shuffle(indices)
    split_indices = np.array_split(indices, N_CLIENTS)
    
    train_sets = []
    cum_sum_train = 0
    for i, conf in enumerate(client_conf):
        local_dataset = include_digits(create_subset(x_train, y_train, split_indices[i]), conf)
        train_sets.append(local_dataset)
        
        cum_sum_train += len(local_dataset)
        print(f"Total examples for client {conf}: {len(local_dataset)}")
    
    print(f"Total examples for all clients: {cum_sum_train}")
        
    testset = list(zip(x_test, y_test))
    
    return train_sets, testset
    
        
def create_subset(x, y, indices):
    """
    Creates a subset of the data based on the provided indices.

    Args:
        x (np.array): Feature data.
        y (np.array): Labels.
        indices (np.array): Indices to select the subset.

    Returns:
        list: A list of tuples representing the subset of data.
    """
    return [(x[i], y[i]) for i in indices]

def build_model():
    """
    Builds and compiles a neural network model using Keras.

    Returns:
        tf.keras.Model: The compiled neural network model.
    """
    model = models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


def train_model(model, train_set, num_epochs=10, batch_size=64):
    """
    Trains the given model on the provided training set.

    Args:
        model (keras.Model): The model to train.
        train_set (list): The training dataset, with features and labels.
        num_epochs (int, optional): Number of training epochs. Defaults to 10.
        batch_size (int, optional): Batch size for training. Defaults to 64.

    Returns:
        keras.callbacks.History: History object containing training metrics.
    """
    x_train, y_train = zip(*train_set)
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
    
    return history


def evaluate_model(model, test_set):
    """
    Evaluates the model on the provided test set.

    Args:
        model (keras.Model): The model to evaluate.
        test_set (list): The test dataset, with features and labels.

    Returns:
        Tuple: Loss and accuracy of the model on the test set.
    """
    x_test, y_test = zip(*test_set)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return loss, accuracy


def include_digits(dataset, included_digits):
    """
    Filters the dataset to include only the specified digits.

    Args:
        dataset (list): The dataset to filter, containing tuples of images and labels.
        included_digits (list): List of digits to include.

    Returns:
        list: Filtered dataset containing only the specified digits.
    """
    filtered_indices = [i for i, (img, label) in enumerate(dataset) if label in included_digits]
    filtered_dataset = [dataset[i] for i in filtered_indices]
    return filtered_dataset


def exclude_digits(dataset, excluded_digits):
    """
    Filters the dataset to exclude the specified digits.

    Args:
        dataset (list): The dataset to filter, containing tuples of images and labels.
        excluded_digits (list): List of digits to exclude.

    Returns:
        list: Filtered dataset excluding the specified digits.
    """
    filtered_indices = [i for i, (img, label) in enumerate(dataset) if label not in excluded_digits]
    filtered_dataset = [dataset[i] for i in filtered_indices]
    return filtered_dataset


def compute_confusion_matrix(model, testset):
    """
    Computes the confusion matrix based on the model's predictions on the test set.

    Args:
        model (keras.Model): The model to use for predictions.
        testset (list): The test dataset, with features and labels.

    Returns:
        np.array: The confusion matrix.
    """
    true_labels = []
    predicted_labels = []

    x_test, y_test = zip(*testset)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = y_test

    cm = confusion_matrix(true_labels, predicted_labels)
    return cm


def plot_confusion_matrix(cm, title):
    """
    Plots a confusion matrix using seaborn.

    Args:
        cm (np.array): The confusion matrix to plot.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", linewidths=0.5)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def plot_learning_curves(history):
    """
    Plots the training and validation accuracy curves with the last epoch's accuracy values in the title.

    Args:
        history: History object returned by Keras during training (from model.fit()).
    """
    # Extract accuracy for training and validation
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Get the last accuracy values
    best_acc = max(val_acc)

    # Create figure for accuracy
    epochs_range = range(1, len(acc) + 1)
    
    plt.figure(figsize=(8, 6))
    
    # Plot accuracy
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')

    # Set title with the last training and validation accuracy
    plt.title(f'Training and Validation Accuracy\n'
              f'Best Model Accuracy: {best_acc:.4f}')
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Show the plot
    plt.show()

backend_setup = {
    "init_args": {
        "logging_level": ERROR, 
        "log_to_driver": False
    }
}


def set_weights(model, parameters):
    """
    Sets the weights of a model using the provided parameters.

    Args:
        model (keras.Model): The model whose weights will be set.
        parameters (list): List of parameters to set as the model's weights.
    """
    model.set_weights([tf.convert_to_tensor(p) for p in parameters])


def get_weights(model):
    """
    Retrieves the weights from the given model.

    Args:
        model (keras.Model): The model to retrieve weights from.

    Returns:
        list: List of weights from the model.
    """
    return model.get_weights()


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Computes the weighted average accuracy across multiple clients.

    Args:
        metrics (List[Tuple[int, Metrics]]): List of tuples, where each tuple contains
                                             the number of examples and the accuracy metrics from each client.

    Returns:
        Metrics: A dictionary containing the weighted average accuracy.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def plot_label_histogram(dataset, title):
    """
    Plots a histogram of label distributions in the dataset with bars centered on digits 0-9.
    
    Parameters:
    dataset (list): A dataset where each entry is a tuple with the label in the second element.
    """
    # Extract the labels (second element in dataset entries)
    labels = [dataset[i][1] for i in range(len(dataset))]

    # Create a histogram with bars centered on 0-9
    plt.hist(labels, bins=range(11), align='left', rwidth=0.8, color='skyblue', edgecolor='black')

    # Add labels and title
    plt.xticks(range(10))  # Ensure the x-ticks are centered on digits 0-9
    plt.xlabel('Digit')
    plt.ylabel('Frequency')
    plt.title(title)

    # Show the plot
    plt.show()
  
    
def plot_combined(cm, history, cm_title, global_title):
    """
    Creates a combined plot with the confusion matrix on the left and learning curves on the right,
    with a global title at the top.

    Args:
        cm: Confusion matrix (as returned by sklearn.metrics.confusion_matrix).
        history: History object returned by Keras during training (from model.fit()).
        cm_title: Title for the confusion matrix.
        global_title: Global title for the entire plot.
    """
    # Extract accuracy for training and validation
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    # Get the best validation accuracy
    best_acc = max(val_acc)

    # Create a figure with two subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot confusion matrix on the left
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", linewidths=0.5, ax=axes[0])
    axes[0].set_title(cm_title)
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Plot learning curves on the right
    epochs_range = range(1, len(acc) + 1)
    axes[1].plot(epochs_range, acc, label='Training Accuracy')
    axes[1].plot(epochs_range, val_acc, label='Validation Accuracy')
    axes[1].set_title(f'Learning curves\nBest Model Accuracy: {best_acc:.4f}')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    # Set global title
    plt.suptitle(global_title, fontsize=16)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect parameter to avoid title overlap

    # Show the combined plot
    plt.show()