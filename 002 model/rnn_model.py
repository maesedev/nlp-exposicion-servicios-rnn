"""
Decoder-only BPE-token RNN for dinosaur name generation.

Architecture (from diagram):
  Embedding → LSTM → Dense(softmax)

Vocabulary and merge rules come from 001 Proprocesamiento/tokens/.
token_encoder.py handles normalization, BPE encoding, and SOW/EOW wrapping.
Training uses teacher forcing: input at step t is the true previous token.
"""

import csv
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPRO_DIR = os.path.join(BASE_DIR, "../001 Proprocesamiento")
VOCAB_PATH = os.path.join(PREPRO_DIR, "tokens/vocabulary.txt")
MERGES_PATH = os.path.join(PREPRO_DIR, "tokens/merges.csv")
DATA_PATH = os.path.join(BASE_DIR, "../000 data/Dinosours.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_model.keras")

sys.path.insert(0, PREPRO_DIR)
from token_encoder import encode, load_merges, load_vocab  # type: ignore  # noqa: E402

# ── Hyperparameters ───────────────────────────────────────────────────────────
HIDDEN_UNITS = 128
EMBEDDING_DIM = 32
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TEST_SPLIT = 0.2
SEED = 42

PAD_IDX = 0  # <PAD> is line 0 in vocabulary.txt → mask_zero=True works


# ── Data ──────────────────────────────────────────────────────────────────────

def load_names(path: str) -> list[str]:
    names = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["nombre"].strip()
            if name:
                names.append(name)
    return names


def make_sequences(
    names: list[str],
    vocab: dict[str, int],
    merges: list[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode each name with BPE (token_encoder handles normalization + SOW/EOW),
    then build teacher-forcing (input, target) pairs:
      encoded : [<SOW>, t0, t1, ..., tn, <EOW>]
      input   : [<SOW>, t0, t1, ..., tn]      ← all except last
      target  : [t0,    t1, ..., tn, <EOW>]   ← all except first

    Right-padded with PAD_IDX (0) so mask_zero=True suppresses padding loss.
    """
    xs, ys = [], []
    unk_idx = vocab["<UNK>"]
    for name in names:
        tokens = encode(name, merges, vocab=vocab)  # returns list of token strings
        ix = [vocab.get(t, unk_idx) for t in tokens]
        xs.append(ix[:-1])
        ys.append(ix[1:])

    max_len = max(len(x) for x in xs)
    X = tf.keras.preprocessing.sequence.pad_sequences(
        xs, maxlen=max_len, padding="post", value=PAD_IDX
    )
    Y = tf.keras.preprocessing.sequence.pad_sequences(
        ys, maxlen=max_len, padding="post", value=PAD_IDX
    )
    return X.astype(np.int32), Y.astype(np.int32)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(vocab_size: int) -> tf.keras.Model:
    """
    Decoder-only LSTM:
      Embedding (mask_zero=True) → LSTM → Dense → Softmax

    The padding mask from Embedding propagates through LSTM so padded
    positions do not contribute to the loss.
    """
    model = tf.keras.Sequential(
        [
            layers.Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True),
            layers.LSTM(HIDDEN_UNITS, return_sequences=True),
            layers.Dense(vocab_size, activation="softmax"),
        ],
        name="decoder_only_rnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train() -> tuple[
    tf.keras.Model,
    dict[str, list[float]],
    dict[str, int],
    dict[str, float],
]:
    """
    Full training pipeline.

    Returns
    -------
    model        : trained Keras model (also saved to MODEL_SAVE_PATH)
    history      : dict with per-epoch metrics (loss, accuracy, val_loss, val_accuracy)
    vocab        : token → int mapping (from token_encoder)
    test_metrics : {'test_loss': float, 'test_accuracy': float}
    """
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    vocab = load_vocab(VOCAB_PATH)
    merges = load_merges(MERGES_PATH)
    vocab_size = len(vocab)

    names = load_names(DATA_PATH)
    X, Y = make_sequences(names, vocab, merges)

    # ── Train / test split ────────────────────────────────────────────────────
    n = len(X)
    split = int(n * (1 - TEST_SPLIT))
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    print(f"Vocabulary size : {vocab_size}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples    : {len(X_test)}")

    # ── Build & train ─────────────────────────────────────────────────────────
    model = build_model(vocab_size)
    model.summary()

    history = model.fit(
        X_train,
        Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, Y_test),
        verbose=1,
    )

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=0)
    test_metrics = {"test_loss": float(test_loss), "test_accuracy": float(test_acc)}

    print(f"\nTest loss     : {test_loss:.4f}")
    print(f"Test accuracy : {test_acc:.4f}")

    # ── Persist artifacts ─────────────────────────────────────────────────────
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved → {MODEL_SAVE_PATH}")

    return model, history.history, vocab, test_metrics


if __name__ == "__main__":
    train()
