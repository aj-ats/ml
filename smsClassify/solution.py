"""
FreeCodeCamp — Neural Network SMS Text Classifier (first-pass solution).

Run from this directory:

    cd smsClassify
    python solution.py

Downloads train-data.tsv / valid-data.tsv if missing.
Pass criterion: predict_message() labels all 7 grader messages correctly.
"""

from __future__ import annotations

import os
from urllib.request import urlretrieve

import matplotlib

if os.environ.get("DISPLAY") is None and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_PATH = "train-data.tsv"
VALID_PATH = "valid-data.tsv"
BASE_URL = "https://cdn.freecodecamp.org/project-data/sms"

MAX_FEATURES = 5000
MAX_LEN = 100
EMBED_DIM = 32
BATCH_SIZE = 32
EPOCHS = 15
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") == "1"

# FreeCodeCamp grader messages / expected labels (do not change)
TEST_MESSAGES = [
    "how are you doing today",
    "sale today! to stop texts call 98912460324",
    "i dont want to go. can we try it a different day? available sat",
    "our new mobile video service is live. just install on your phone to start watching.",
    "you have won £1000 cash! call to claim your prize.",
    "i'll bring it tomorrow. don't forget the milk.",
    "wow, is your arm alright. that happened to me one time too",
]
TEST_ANSWERS = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def ensure_data() -> None:
    for path in (TRAIN_PATH, VALID_PATH):
        if not os.path.isfile(path):
            print(f"Downloading {path}...")
            urlretrieve(f"{BASE_URL}/{path}", path)


def load_tsv(path: str) -> pd.DataFrame:
    """Load label\\tmessage TSV with no header."""
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "message"])
    # ham=0, spam=1  →  sigmoid = P(spam)
    df["label_id"] = df["label"].map({"ham": 0, "spam": 1}).astype("float32")
    return df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(vectorize_layer: layers.TextVectorization) -> keras.Model:
    """
    string → TextVectorization → Embedding → GAP → Dense → P(spam).

    Vectorizer is part of the model so train and predict share the same path.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(), dtype=tf.string),
            vectorize_layer,
            layers.Embedding(MAX_FEATURES, EMBED_DIM),
            layers.GlobalAveragePooling1D(),
            layers.Dense(24, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Predict API (required by the challenge)
# ---------------------------------------------------------------------------
# Filled in main() after training so predict_message can close over model.
_model: keras.Model | None = None


def predict_message(pred_text: str) -> list:
    """
    Return [probability, label] for one SMS string.

    Example: [0.0083, 'ham']
    Probability is P(spam); label is 'spam' if P > 0.5 else 'ham'.
    """
    if _model is None:
        raise RuntimeError("Model not trained yet. Call main() first.")
    prob = float(_model.predict(tf.constant([pred_text]), verbose=0)[0][0])
    label = "spam" if prob > 0.5 else "ham"
    return [prob, label]


def test_predictions() -> bool:
    """Same checks as the notebook grader cell."""
    passed = True
    for msg, ans in zip(TEST_MESSAGES, TEST_ANSWERS):
        prediction = predict_message(msg)
        ok = prediction[1] == ans
        status = "OK" if ok else "FAIL"
        print(
            f"  [{status}] {prediction[1]!r:5s} (p={prediction[0]:.4f})  "
            f"expected={ans!r}  | {msg[:50]}"
        )
        if not ok:
            passed = False
    if passed:
        print("You passed the challenge. Great job!")
    else:
        print("You haven't passed yet. Keep trying.")
    return passed


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def plot_history(
    history: keras.callbacks.History, path: str = "sms_training_curves.png"
) -> None:
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="train acc")
    plt.plot(epochs_range, val_acc, label="val acc")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="train loss")
    plt.plot(epochs_range, val_loss, label="val loss")
    plt.legend()
    plt.title("Loss")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    print("Saved", path)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def main() -> None:
    global _model

    print("TensorFlow", tf.__version__)
    ensure_data()

    train_df = load_tsv(TRAIN_PATH)
    valid_df = load_tsv(VALID_PATH)

    print(train_df["label"].value_counts().to_string())
    print(f"train={len(train_df)}, valid={len(valid_df)}")

    vectorize_layer = layers.TextVectorization(
        max_tokens=MAX_FEATURES,
        output_mode="int",
        output_sequence_length=MAX_LEN,
    )
    vectorize_layer.adapt(train_df["message"].values)

    train_texts = train_df["message"].values
    train_labels = train_df["label_id"].values
    val_texts = valid_df["message"].values
    val_labels = valid_df["label_id"].values

    model = build_model(vectorize_layer)
    model.summary()

    # Class weights: dataset is ~86% ham — without this the model may ignore spam
    n_ham = int((train_labels == 0).sum())
    n_spam = int((train_labels == 1).sum())
    total = n_ham + n_spam
    class_weight = {
        0: total / (2.0 * n_ham),
        1: total / (2.0 * n_spam),
    }
    print("class_weight:", class_weight)

    history = model.fit(
        train_texts,
        train_labels,
        validation_data=(val_texts, val_labels),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        verbose=1,
    )

    val_loss, val_acc = model.evaluate(val_texts, val_labels, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}  loss: {val_loss:.4f}")
    plot_history(history)

    _model = model

    print("\nSample:", predict_message("how are you doing today?"))
    print("\nGrader:")
    test_predictions()


if __name__ == "__main__":
    main()
