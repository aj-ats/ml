from __future__ import annotations
 
import os
import zipfile
from urllib.request import urlretrieve
 
import matplotlib
 
# Headless-friendly default; still works with a display.
if os.environ.get("DISPLAY") is None and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
 
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PATH = "./cats_and_dogs"
ZIP_PATH = "cats_and_dogs.zip"
URL = "https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip"
 
BATCH_SIZE = 128
EPOCHS = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") == "1"
 
# Fixed FreeCodeCamp test labels (1=dog, 0=cat), same order as test_data_gen
ANSWERS = [
    1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
    1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
    1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
    1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1,
    0, 0, 0, 0, 0, 0,
]
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dataset() -> None:
    """Download/extract cats_and_dogs if the folder is missing."""
    if os.path.isdir(PATH):
        return
    if not os.path.isfile(ZIP_PATH):
        print("Downloading dataset...")
        urlretrieve(URL, ZIP_PATH)
    print("Extracting", ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(".")
 
 
def count_files(directory: str) -> int:
    return sum(len(files) for _, _, files in os.walk(directory))
 
 
def plot_images(images_arr, probabilities=False, save_path: str | None = None) -> None:
    fig, axes = plt.subplots(len(images_arr), 1, figsize=(5, len(images_arr) * 3))
    if len(images_arr) == 1:
        axes = [axes]
    if probabilities is False:
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis("off")
    else:
        for img, probability, ax in zip(images_arr, probabilities, axes):
            ax.imshow(img)
            ax.axis("off")
            if probability > 0.5:
                ax.set_title("%.2f" % (probability * 100) + "% dog")
            else:
                ax.set_title("%.2f" % ((1 - probability) * 100) + "% cat")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print("Saved", save_path)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
 
 
def plot_history(history, save_path: str = "training_curves.png") -> None:
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(len(acc))
 
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")
 
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print("Saved", save_path)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
 
 
def build_model() -> Sequential:
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dropout(0.5),
            Dense(512, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("TensorFlow", tf.__version__)
    ensure_dataset()
 
    train_dir = os.path.join(PATH, "train")
    validation_dir = os.path.join(PATH, "validation")
    test_dir = os.path.join(PATH, "test")
 
    total_train = count_files(train_dir)
    total_val = count_files(validation_dir)
    total_test = len(
        [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    print(f"train={total_train}, val={total_val}, test={total_test}")
 
    # Generators: train with augmentation; val/test only rescale
    train_image_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    validation_image_generator = ImageDataGenerator(rescale=1.0 / 255)
    test_image_generator = ImageDataGenerator(rescale=1.0 / 255)
 
    train_data_gen = train_image_generator.flow_from_directory(
        batch_size=BATCH_SIZE,
        directory=train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode="binary",
    )
    val_data_gen = validation_image_generator.flow_from_directory(
        batch_size=BATCH_SIZE,
        directory=validation_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode="binary",
    )
    # Flat test folder: use PATH + classes=["test"], keep order for grader
    test_data_gen = test_image_generator.flow_from_directory(
        batch_size=BATCH_SIZE,
        directory=PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        classes=["test"],
        class_mode="binary",
        shuffle=False,
    )
 
    model = build_model()
    model.summary()
 
    history = model.fit(
        train_data_gen,
        steps_per_epoch=total_train // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_data_gen,
        validation_steps=total_val // BATCH_SIZE,
    )
    plot_history(history)
 
    # Predictions: P(dog) for each test image
    test_data_gen.reset()
    pred = model.predict(
        test_data_gen,
        steps=int(np.ceil(total_test / float(BATCH_SIZE))),
    )
    probabilities = pred.flatten().tolist()
 
    print("Sample probabilities (first 10):", [round(p, 3) for p in probabilities[:10]])
    print(f"Predicted {len(probabilities)} images.")
 
    # Optional: save a small sample plot of first 5 test images
    test_data_gen.reset()
    batch = next(test_data_gen)
    imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
    plot_images(imgs[:5], probabilities[:5], save_path="test_sample.png")
 
    # Challenge check
    correct = sum(
        1
        for probability, answer in zip(probabilities, ANSWERS)
        if round(probability) == answer
    )
    percentage_identified = (correct / len(ANSWERS)) * 100
    passed = percentage_identified >= 63
 
    print(
        f"Your model correctly identified {round(percentage_identified, 2)}% "
        "of the images of cats and dogs."
    )
    if passed:
        print("You passed the challenge!")
    else:
        print(
            "You haven't passed yet. Your model should identify at least 63% "
            "of the images. Keep trying. You will get it!"
        )
 
 
if __name__ == "__main__":
    main()
