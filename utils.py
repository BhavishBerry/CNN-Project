import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
import gc
from tensorflow.keras import backend as K
import tensorflow as tf
import zipfile
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

# ----------------------------- Existing Functions (with docstrings added) -----------------------------

def plot_loss_accuracy(history):
    """
    Plots the training and validation loss and accuracy curves.

    Args:
        history: TensorFlow History object from model.fit().
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axs[0].plot(history.history['loss'], label='Train Loss')
    axs[0].plot(history.history['val_loss'], label='Val Loss')
    axs[0].set_title('Loss Curve')
    axs[0].legend()

    # Accuracy
    axs[1].plot(history.history['accuracy'], label='Train Acc')
    axs[1].plot(history.history['val_accuracy'], label='Val Acc')
    axs[1].set_title('Accuracy Curve')
    axs[1].legend()

    plt.show()

def save_model_summary(model, filename="model_summary.txt"):
    """
    Saves the summary of a model to a text file.

    Args:
        model: Keras model instance.
        filename (str): Output filename. Default is "model_summary.txt".
    """
    with open(filename, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

def get_model_size(model):
    """
    Returns the on-disk size of a Keras model in megabytes.

    Args:
        model: Keras model instance.

    Returns:
        str: Model size in MB.
    """
    model_file = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
    model.save(model_file.name)
    size = os.path.getsize(model_file.name) / (1024 ** 2)
    model_file.close()
    os.remove(model_file.name)
    return f"{size:.2f} MB"

def plot_predictions(model, data, num_samples=30):
    """
    Plots prediction results from a model on a batch of images.

    Args:
        model: A trained Keras model.
        data: A Keras DirectoryIterator object.
        num_samples (int): Number of samples to plot.
    """
    plt.figure(figsize=(20, 20))
    images_plotted = 0
    idx_to_class = {v: k for k, v in data.class_indices.items()}

    for batch_images, batch_labels in data:
        for i in range(len(batch_images)):
            if images_plotted >= num_samples:
                break

            sample_image = batch_images[i]
            sample_label = batch_labels[i]
            true_label_idx = np.argmax(sample_label)
            true_label_name = idx_to_class[true_label_idx]

            pred_probs = model.predict(tf.expand_dims(sample_image, axis=0), verbose=0)
            pred_label_idx = np.argmax(pred_probs)
            pred_label_name = idx_to_class[pred_label_idx]

            plt.subplot(5, 6, images_plotted + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(f"Actual: {true_label_name}\nPredicted: {pred_label_name}")
            plt.imshow(sample_image)

            images_plotted += 1
        if images_plotted >= num_samples:
            break

def reset_gpu():
    """
    Frees up GPU memory by clearing Keras backend session and collecting garbage.
    """
    K.clear_session()
    gc.collect()

# ----------------------------- New & Enhanced Functions -----------------------------

def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Loads and preprocesses an image from a file path.

    Args:
        filename (str): Path to the image file.
        img_shape (int): Target width and height. Default is 224.
        scale (bool): Whether to scale pixel values to 0–1. Default is True.

    Returns:
        tf.Tensor: Preprocessed image tensor.
    """
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_shape, img_shape])
    return img / 255. if scale else img

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=12, norm=False, savefig=False):
    """
    Plots a styled confusion matrix using true and predicted labels.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        classes (list): Optional list of class names.
        figsize (tuple): Figure size.
        text_size (int): Font size inside matrix.
        norm (bool): Whether to normalize values.
        savefig (bool): Whether to save figure as "confusion_matrix.png".
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)

    labels = classes if classes else np.arange(n_classes)
    ax.set(title="Confusion Matrix",
           xlabel="Predicted Label",
           ylabel="True Label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)

    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        value = f"{cm[i, j]}" if not norm else f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)"
        ax.text(j, i, value, ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black", fontsize=text_size)

    if savefig:
        fig.savefig("confusion_matrix.png")

def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Plots combined training history before and after fine-tuning.

    Args:
        original_history: History object from initial training.
        new_history: History object from fine-tuning.
        initial_epochs (int): Number of epochs in the original history.
    """
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]
    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]
    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    plt.figure(figsize=(8, 8))

    # Accuracy
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Train Accuracy')
    plt.plot(total_val_acc, label='Val Accuracy')
    plt.axvline(x=initial_epochs-1, color='gray', linestyle='--', label='Start Fine-tuning')
    plt.legend()
    plt.title("Accuracy Over Time")

    # Loss
    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Train Loss')
    plt.plot(total_val_loss, label='Val Loss')
    plt.axvline(x=initial_epochs-1, color='gray', linestyle='--', label='Start Fine-tuning')
    plt.legend()
    plt.title("Loss Over Time")
    plt.xlabel("Epochs")
    plt.show()

def unzip_data(filename):
    """
    Unzips a .zip file to the current working directory.

    Args:
        filename (str): Path to .zip file.
    """
    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall()
    zip_ref.close()

def walk_through_dir(dir_path):
    """
    Walks through a directory and prints the number of files in each subdirectory.

    Args:
        dir_path (str): Path to the base directory.
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def calculate_results(y_true, y_pred):
    """
    Calculates accuracy, precision, recall, and F1-score.

    Args:
        y_true (list or array): Ground truth labels.
        y_pred (list or array): Predicted labels.

    Returns:
        dict: Dictionary with accuracy, precision, recall, and f1-score.
    """
    acc = accuracy_score(y_true, y_pred) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
