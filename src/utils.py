import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import itertools
from sklearn.metrics import confusion_matrix

# Function to setup logger
def setup_logger(output_path):
    os.makedirs(output_path, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)
    
    # Log file handler - log all messages and timestamp
    file_handler = logging.FileHandler(os.path.join(output_path, 'training.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
    
    # Console handler - log only the message
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# Plot confusion matrix function
def plot_confusion_matrix(true_labels, predictions, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(true_labels, predictions)
    # Compute overall accuracy, class recall and precision, F1 score
    overall_accuracy = np.trace(cm) / float(np.sum(cm))

    # Calculate accuracy with allowed confusion, only for 6 classes of Malaria dataset
    relaxed_accuracy = 0
    if len(classes) == 6:
        total_samples = np.sum(cm)
        correct_with_tolerance = 0
        # Define allowed confusion ranges for each class
        tolerance_ranges = {
            0: [0, 1],      # Class 1 allows confusion with 1,2
            1: [0, 1, 2],   # Class 2 allows confusion with 1,2,3
            2: [1, 2, 3],   # Class 3 allows confusion with 2,3,4
            3: [2, 3],      # Class 4 allows confusion with 3,4
            4: [4],      # Class 5
            5: [5]       # Class 6
        }
        for true_class in range(len(classes)):
            allowed_predictions = tolerance_ranges[true_class]
            for pred_class in allowed_predictions:
                correct_with_tolerance += cm[true_class, pred_class]
        relaxed_accuracy = correct_with_tolerance / total_samples

    # class_recall = cm.diagonal() / cm.sum(axis=1)
    # class_precision = cm.diagonal() / cm.sum(axis=0)
    # F1_score_micro = 2 * np.sum(class_precision * class_recall) / np.sum(class_precision + class_recall)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # Plot accuracy and F1 score
    # plt.title(f'{title} - Overall Accuracy: {overall_accuracy:.4f}')
    plt.title(f'Overall Acc: {overall_accuracy:.4f}, Relaxed Acc: {relaxed_accuracy:.4f}')

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Plot confusion matrix values
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    return relaxed_accuracy

def plot_results(train_losses, val_losses, train_accuracies, val_accuracies, output_path):
    # Plot and save loss and accuracy curves
    plt.figure(figsize=(12, 12))
    plt.subplot(2, 2, 1) 
    plt.plot(train_losses, label='Train Loss')
    plt.title(f'Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.title(f'Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title(f'Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(output_path, f'loss_accuracy.png'))
    plt.show()

def visualize_d_matrix(original, d_matrix, save_path):
    """
    Save the original image and its corresponding D matrix channels without normalization
    """
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    
    # Plot original image in RGB
    axes[0, 0].imshow(original.permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")
    
    # Plot D-matrix channels directly
    for i in range(3):
        ax = axes[(i+1)//2, (i+1)%2]
        d_channel = d_matrix[i].cpu().numpy()
        
        # 
        im = ax.imshow(d_channel, cmap='gray')
        ax.set_title(f"D-matrix Channel {i+1}")
        ax.axis("off")
        
        # 
        # plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
