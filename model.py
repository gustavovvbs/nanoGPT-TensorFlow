import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    LayerNormalization,
    Dropout,
    TextVectorization,
)
from tensorflow.keras.models import Model
import time

# -------------------------------
# Configuration and Hyperparameters
# -------------------------------

print("Available GPU devices:", tf.config.list_physical_devices("GPU"))

# Model hyperparameters
EMBED_DIM = 512          # Dimension of token embeddings
NUM_HEADS = 8            # Number of attention heads
BLOCK_SIZE = 128         # Context size (sequence length)
HEAD_DIM = EMBED_DIM // NUM_HEADS  # Dimension per attention head
VOCAB_SIZE = 100         # Maximum vocabulary size (will be updated after preprocessing)
NUM_BLOCKS = 6           # Number of transformer blocks
DROPOUT_RATE = 0.1       # Dropout rate for regularization

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.9

SHUFFLE_BUFFER_SIZE = 10000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------

def load_text_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def preprocess_text(text, max_tokens=None):
    vectorizer = TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        split='character',  # Split text into individual characters
        standardize=None,   # Do not standardize (e.g., lowercasing)
    )
    vectorizer.adapt([text])

    encoded_text = vectorizer([text])  # (1, sequence_length)
    encoded_text = tf.squeeze(encoded_text, axis=0)  #(sequence_length,)

    vocab_size = len(vectorizer.get_vocabulary())

    return encoded_text, vocab_size, vectorizer

data_file = "data.txt"
print(f"Loading data from {data_file}...")
raw_text = load_text_data(data_file)
encoded_text, VOCAB_SIZE, vectorizer = preprocess_text(raw_text, max_tokens=VOCAB_SIZE)

print(encoded_text)
split_index = int(encoded_text.shape[0] * TRAIN_SPLIT)
train_sequence = encoded_text[:split_index]
test_sequence = encoded_text[split_index:]

print(f"Total tokens: {len(encoded_text)}")
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Training tokens: {len(train_sequence)}")
print(f"Testing tokens: {len(test_sequence)}")

# -------------------------------
# Dataset Preparation
# -------------------------------

def create_dataset(sequence, sequence_length, batch_size, shuffle=False, seed=None):
    dataset = tf.data.Dataset.from_tensor_slices(sequence)

    sequences = dataset.batch(sequence_length + 1, drop_remainder=True)

    if shuffle:
        sequences = sequences.shuffle(SHUFFLE_BUFFER_SIZE, seed=seed)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(PREFETCH_BUFFER_SIZE)

    return dataset

train_dataset = create_dataset(
    train_sequence,
    sequence_length=BLOCK_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
)
test_dataset = create_dataset(
    test_sequence,
    sequence_length=BLOCK_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# -------------------------------
# Model Components
# -------------------------------

class MultiHeadAttention(Model):
    def __init__(self, num_heads, head_dim, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.all_head_dim = num_heads * head_dim

        self.qkv = Dense(self.all_head_dim * 3, use_bias=False, name="qkv")
        self.out_proj = Dense(self.all_head_dim, use_bias=False, name="out_proj")
        self.dropout = Dropout(DROPOUT_RATE)

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        qkv = self.qkv(x)  # (batch_size, seq_length, all_head_dim * 3)
        qkv = tf.reshape(qkv, [batch_size, seq_length, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])  # (3, batch_size, num_heads, seq_length, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scaling = float(self.head_dim) ** -0.5
        q = q * scaling

        attention_scores = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_length, seq_length)

        mask = tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)
        mask = tf.cast(mask, dtype=tf.float32)
        attention_scores = attention_scores * mask - 1e10 * (1 - mask)

        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)

        attention_output = tf.matmul(attention_probs, v)  # (batch_size, num_heads, seq_length, head_dim)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])  # (batch_size, seq_length, num_heads, head_dim)
        attention_output = tf.reshape(attention_output, [batch_size, seq_length, self.all_head_dim])

        output = self.out_proj(attention_output)
        return output

class FeedForward(Model):
    def __init__(self, embed_dim, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.dense1 = Dense(embed_dim * 4, activation='gelu', name="ff_dense1")
        self.dense2 = Dense(embed_dim, name="ff_dense2")
        self.dropout = Dropout(DROPOUT_RATE)

    def call(self, x, training):
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

class TransformerBlock(Model):
    def __init__(self, num_heads, head_dim, embed_dim, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.layer_norm1 = LayerNormalization(epsilon=1e-5, name="ln1")
        self.mha = MultiHeadAttention(num_heads=num_heads, head_dim=head_dim, name="multi_head_attention")
        self.layer_norm2 = LayerNormalization(epsilon=1e-5, name="ln2")
        self.ffn = FeedForward(embed_dim=embed_dim, name="feed_forward")

    def call(self, x, training):
        residual = x
        x = self.layer_norm1(x)
        x = self.mha(x, training=training)
        x = residual + x

        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x, training=training)
        x = residual + x
        return x

class NanoGPT(Model):
    def __init__(self, vocab_size, embed_dim, num_heads, num_blocks, block_size, **kwargs):
        super(NanoGPT, self).__init__(**kwargs)
        self.token_embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim, name="token_embedding")
        self.position_embedding = Embedding(input_dim=block_size, output_dim=embed_dim, name="position_embedding")
        self.dropout = Dropout(DROPOUT_RATE)
        self.blocks = [TransformerBlock(num_heads=num_heads, head_dim=embed_dim // num_heads, embed_dim=embed_dim, name=f"block_{i}") for i in range(num_blocks)]
        self.layer_norm = LayerNormalization(epsilon=1e-5, name="ln_f")
        self.head = Dense(vocab_size, name="head")

    def call(self, idx, training):
        batch_size = tf.shape(idx)[0]
        seq_length = tf.shape(idx)[1]
        token_embeddings = self.token_embedding(idx)  # (batch_size, seq_length, embed_dim)
        positions = tf.range(seq_length)
        position_embeddings = self.position_embedding(positions)[tf.newaxis, ...]  # (1, seq_length, embed_dim)
        x = token_embeddings + position_embeddings
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.layer_norm(x)
        logits = self.head(x)
        return logits

    def generate(self, input_ids, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -BLOCK_SIZE:]
            logits = self(idx_cond, training=False)
            logits = logits[:, -1, :]  # (batch_size, vocab_size)
            next_token = tf.argmax(logits, axis=-1)
            next_token = next_token[:, tf.newaxis]  # (batch_size, 1)
            input_ids = tf.concat([input_ids, next_token], axis=1)
        return input_ids

def decode_tokens(token_ids, vectorizer):
    vocab = vectorizer.get_vocabulary()
    tokens = [vocab[token_id] for token_id in token_ids.numpy()]
    return "".join(tokens)

def generate_text(model, start_string, vectorizer, max_new_tokens=100):
    input_ids = vectorizer([start_string])[0]
    input_ids = tf.expand_dims(input_ids, 0)  # (1, time)
    generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    generated_text = decode_tokens(generated_ids[0], vectorizer)
    return generated_text

# -------------------------------
# Model Compilation and Training
# -------------------------------

print("Initializing NanoGPT model...")
gpt_model = NanoGPT(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_blocks=NUM_BLOCKS,
    block_size=BLOCK_SIZE,
)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

train_loss_metric = tf.keras.metrics.Mean()

def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = gpt_model(x, training=True)  # (batch_size, seq_length, vocab_size)
        loss = loss_fn(y, logits)
        loss += sum(gpt_model.losses)

    gradients = tape.gradient(loss, gpt_model.trainable_variables)
    gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
    optimizer.apply_gradients(zip(gradients, gpt_model.trainable_variables))
    train_loss_metric.update_state(loss)
    mean_norm = sum([tf.norm(g).numpy() for g in gradients]) / len(gradients)

    return loss, mean_norm

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    start_time = time.time()
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        B, T = x_batch_train.shape
        step_start = time.time()
        loss, norm = train_step(x_batch_train, y_batch_train)
        step_end = time.time()
        if step % 10 == 0:
            print(f"Step {step}, Loss: {train_loss_metric.result().numpy():.4f}, norm: {norm}, token/s: {B*T/(step_end - step_start)}")

    generated_text = generate_text(gpt_model, "Once upon a time", vectorizer, max_new_tokens=100)
    print(f"\nGenerated Text after Epoch {epoch + 1}:\n{generated_text}\n")

    time_taken = time.time() - start_time
    print(f"Time taken for epoch {epoch + 1}: {time_taken:.2f} secs\n")

gpt_model.save('nano_gpt_model')