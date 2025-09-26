"""
Visualization Module for DINOv3 Training

This module contains functions for plotting and visualizing training results.

Author: GitHub Copilot
Date: 2025-09-13
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(results, figsize=(15, 5)):
    """
    Plot training history including loss and accuracy curves.
    
    Parameters:
    -----------
    results : dict
        Training results containing 'train_losses', 'val_losses', 'train_accs', 'val_accs'
    figsize : tuple, default=(15, 5)
        Figure size for the plots
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot losses
    if 'train_losses' in results and 'val_losses' in results:
        epochs = range(1, len(results['train_losses']) + 1)
        axes[0].plot(epochs, results['train_losses'], 'b-', label='Training Loss')
        axes[0].plot(epochs, results['val_losses'], 'r-', label='Validation Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
    
    # Plot accuracies
    if 'train_accs' in results and 'val_accs' in results:
        epochs = range(1, len(results['train_accs']) + 1)
        axes[1].plot(epochs, results['train_accs'], 'b-', label='Training Accuracy')
        axes[1].plot(epochs, results['val_accs'], 'r-', label='Validation Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print final metrics
    if 'best_val_acc' in results:
        print(f"Best validation accuracy: {results['best_val_acc']:.4f}")
    if 'epochs_trained' in results:
        print(f"Total epochs trained: {results['epochs_trained']}")


def plot_class_distribution(labels, title="Class Distribution"):
    """
    Plot distribution of classes in the dataset.
    
    Parameters:
    -----------
    labels : numpy.ndarray
        Array of class labels
    title : str, default="Class Distribution"
        Title for the plot
    """
    unique_classes, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(unique_classes, counts)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(counts),
                str(count), ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print statistics
    total = len(labels)
    print(f"Total samples: {total}")
    for class_id, count in zip(unique_classes, counts):
        percentage = count / total * 100
        print(f"Class {class_id}: {count} samples ({percentage:.1f}%)")


def plot_sample_predictions(images, true_labels, predictions, confidence_scores=None, 
                          num_samples=6, figsize=(15, 10)):
    """
    Plot sample images with their true labels, predictions, and confidence scores.
    
    Parameters:
    -----------
    images : numpy.ndarray
        Array of images
    true_labels : numpy.ndarray
        True class labels
    predictions : numpy.ndarray
        Predicted class labels
    confidence_scores : numpy.ndarray, optional
        Confidence scores for predictions
    num_samples : int, default=6
        Number of samples to display
    figsize : tuple, default=(15, 10)
        Figure size
    """
    num_samples = min(num_samples, len(images))
    
    if confidence_scores is not None:
        fig, axes = plt.subplots(2, num_samples, figsize=figsize)
        if num_samples == 1:
            axes = axes.reshape(2, 1)
    else:
        fig, axes = plt.subplots(1, num_samples, figsize=(figsize[0], figsize[1]//2))
        if num_samples == 1:
            axes = [axes]
    
    for i in range(num_samples):
        # Main image plot
        row_idx = 0 if confidence_scores is None else 0
        ax = axes[row_idx, i] if confidence_scores is not None else axes[i]
        
        if len(images.shape) == 4:  # Batch of images
            ax.imshow(images[i], cmap='gray')
        else:  # Single image or different format
            ax.imshow(images[i], cmap='gray')
            
        # Determine if prediction is correct
        is_correct = true_labels[i] == predictions[i]
        title_color = 'green' if is_correct else 'red'
        
        title = f'True: {true_labels[i]}, Pred: {predictions[i]}'
        if confidence_scores is not None:
            title += f'\nConf: {confidence_scores[i]:.3f}'
            
        ax.set_title(title, color=title_color)
        ax.axis('off')
        
        # If confidence scores available, show confidence map
        if confidence_scores is not None and len(confidence_scores.shape) > 1:
            axes[1, i].imshow(confidence_scores[i], cmap='viridis', vmin=0, vmax=1)
            axes[1, i].set_title(f'Confidence Map')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print accuracy
    correct = (true_labels[:num_samples] == predictions[:num_samples]).sum()
    accuracy = correct / num_samples
    print(f"Accuracy on displayed samples: {accuracy:.4f} ({correct}/{num_samples})")
