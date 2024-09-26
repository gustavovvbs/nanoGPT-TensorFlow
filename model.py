import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    LayerNormalization,
    Dropout,
    ReLU,
    TextVectorization,
)
from tensorflow.keras.models import Model, Sequential

# -------------------------------
# Configuration and Hyperparameters
# -------------------------------

# Display available GPU devices
print("Available GPU devices:", tf.config.list_physical_devices("GPU"))

# Model hyperparameters
EMBED_DIM = 512          # Dimension of token embeddings
NUM_HEADS = 16           # Number of attention heads
BLOCK_SIZE = 8           # Context size (sequence length)
HEAD_DIM = EMBED_DIM // NUM_HEADS  # Dimension per attention head
VOCAB_SIZE = 100         # Initial vocabulary size 
NUM_BLOCKS = 6           # Number of transformer blocks
DROPOUT_RATE = 0.2       # Dropout rate for regularization

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.9        

SHUFFLE_BUFFER_SIZE = 10000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------

def load_text_data(file_path):
    """
    Load text data from a specified file.

    Args:
        file_path (str): Path to the text file.

    Returns:
        str: Content of the text file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def preprocess_text(text, max_tokens=None):
    """
    Preprocess the text by creating a vocabulary and encoding the text.

    Args:
        text (str): Raw text data.
        max_tokens (int, optional): Maximum number of tokens in the vocabulary.

    Returns:
        tuple: (encoded_text, vocab_size, vectorizer)
    """

    vocab = sorted(set(text))
    vocab_size = len(vocab)
    if max_tokens:
        vocab = vocab[:max_tokens]
        vocab_size = len(vocab)

    vectorizer = TextVectorization(max_tokens=vocab_size, output_mode="int")
    vectorizer.adapt(text)

    # encode to sequence of tokens
    encoded_text = vectorizer(text)

    return encoded_text, vocab_size, vectorizer

data_file = "data.txt"
print(f"Loading data from {data_file}...")
raw_text = load_text_data(data_file)
encoded_text, VOCAB_SIZE, vectorizer = preprocess_text(raw_text)

split_index = int(encoded_text.shape[0]* TRAIN_SPLIT)
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
    """
    Convert a sequence of tokens into a TensorFlow dataset.

    Args:
        sequence (np.ndarray): Array of token IDs.
        sequence_length (int): Length of each input sequence.
        batch_size (int): Number of samples per batch.
        shuffle (bool, optional): Whether to shuffle the dataset.
        seed (int, optional): Random seed for shuffling.

    Returns:
        tf.data.Dataset: Prepared dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices(sequence)

    # Create sliding windows of size `sequence_length + 1`
    dataset = dataset.window(sequence_length + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(sequence_length + 1))

    if shuffle:
        dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE, seed=seed)

    # Split into input and target sequences
    dataset = dataset.map(
        lambda window: (window[:-1], window[1:]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

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

class AttentionHead(Model):
    """
    Single attention head computing self-attention.

    Attributes:
        key (Dense): Dense layer to compute key vectors.
        query (Dense): Dense layer to compute query vectors.
        value (Dense): Dense layer to compute value vectors.
        proj (Dense): Dense layer to project the concatenated outputs.
    """
    def __init__(self, head_size, embed_dim, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)
        self.key = Dense(head_size, name="key")
        self.query = Dense(head_size, name="query")
        self.value = Dense(head_size, name="value")
        self.proj = Dense(embed_dim, name="proj")

    def call(self, x):
        """
        Forward pass for the attention head.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, time, embed_dim).

        Returns:
            tf.Tensor: Output tensor after attention and projection.
        """
        # Compute query, key, and value vectors
        q = self.query(x)  # (batch, time, head_size)
        k = self.key(x)    # (batch, time, head_size)
        v = self.value(x)  # (batch, time, head_size)

        # Compute scaled dot-product attention scores
        attention_scores = tf.matmul(q, k, transpose_b=True)  # (batch, time, time)
        scaling_factor = tf.math.sqrt(tf.cast(HEAD_DIM, tf.float32))
        attention_scores /= scaling_factor

        # causal mask to ensure attention is only to past tokens
        causal_mask = tf.linalg.band_part(tf.ones((tf.shape(x)[1], tf.shape(x)[1])), -1, 0)
        attention_scores = tf.where(
            causal_mask == 0, tf.fill(tf.shape(attention_scores), -1e9), attention_scores
        )

        attention_weights = tf.nn.softmax(attention_scores, axis=-1)  # (batch, time, time)

        # attention output
        attention_output = tf.matmul(attention_weights, v)  # (batch, time, head_size)

        # Projection layer to the skip connection
        output = self.proj(attention_output)  # (batch, time, embed_dim)

        return output

class MultiHeadAttention(Model):
    """
    Multi-head attention layer consisting of multiple attention heads.

    Attributes:
        heads (list of AttentionHead): List of attention heads.
        concat_proj (Dense): Dense layer to project concatenated heads.
        dropout (Dropout): Dropout layer for regularization.
    """
    def __init__(self, num_heads, head_size, embed_dim, dropout_rate=0.2, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = [
            AttentionHead(head_size=head_size, embed_dim=embed_dim, name=f"head_{i}")
            for i in range(num_heads)
        ]
        self.concat_proj = Dense(embed_dim, name="concat_proj")
        self.dropout = Dropout(dropout_rate)

    def call(self, x):
        """
        Forward pass for multi-head attention.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, time, embed_dim).

        Returns:
            tf.Tensor: Output tensor after multi-head attention.
        """
        # Concatenate outputs from all attention heads
        head_outputs = [head(x) for head in self.heads]  # List of len(heads) (batch, time, embed_dim)
        concatenated = tf.concat(head_outputs, axis=-1)  # (batch, time, embed_dim * num_heads)

        # Project the concatenated outputs and apply dropout
        projected = self.concat_proj(concatenated)  # (batch, time, embed_dim)
        output = self.dropout(projected)

        return output

class FeedForward(Model):
    """
    Fully connected feed-forward network with ReLU activation.

    Attributes:
        dense1 (Dense): First dense layer expanding the embedding dimension.
        activation (ReLU): ReLU activation layer.
        dropout (Dropout): Dropout layer for regularization.
        dense2 (Dense): Second dense layer projecting back to embedding dimension.
    """
    def __init__(self, embed_dim, dropout_rate=0.2, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.dense1 = Dense(embed_dim * 4, name="ff_dense1")
        self.activation = ReLU(name="ff_relu")
        self.dropout = Dropout(dropout_rate)
        self.dense2 = Dense(embed_dim, name="ff_dense2")

    def call(self, x):
        """
        Forward pass for the feed-forward network.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, time, embed_dim).

        Returns:
            tf.Tensor: Output tensor after feed-forward operations.
        """
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

class TransformerBlock(Model):
    """
    Single transformer block consisting of multi-head attention and feed-forward network.

    Attributes:
        layer_norm1 (LayerNormalization): Layer normalization before attention.
        multi_head_attention (MultiHeadAttention): Multi-head attention layer.
        layer_norm2 (LayerNormalization): Layer normalization before feed-forward.
        feed_forward (FeedForward): Feed-forward network.
    """
    def __init__(self, num_heads, head_size, embed_dim, dropout_rate=0.2, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.layer_norm1 = LayerNormalization(epsilon=1e-6, name="ln1")
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_heads,
            head_size=head_size,
            embed_dim=embed_dim,
            dropout_rate=dropout_rate,
            name="multi_head_attention",
        )
        self.layer_norm2 = LayerNormalization(epsilon=1e-6, name="ln2")
        self.feed_forward = FeedForward(embed_dim=embed_dim, dropout_rate=dropout_rate, name="feed_forward")

    def call(self, x):
        """
        Forward pass for the transformer block.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, time, embed_dim).

        Returns:
            tf.Tensor: Output tensor after transformer block.
        """
        # mha with skip connection
        attention_output = self.multi_head_attention(self.layer_norm1(x))
        x = x + attention_output

        # mlp with skip connection
        ff_output = self.feed_forward(self.layer_norm2(x))
        x = x + ff_output

        return x

class NanoGPT(Model):
    """
    A minimalist GPT-like model.

    Attributes:
        token_embedding (Embedding): Embedding layer for tokens.
        position_embedding (Embedding): Embedding layer for positional encoding.
        transformer_blocks (Sequential): Stack of transformer blocks.
        layer_norm (LayerNormalization): Final layer normalization.
        logits_dense (Dense): Dense layer to produce logits for each token in the vocabulary.
    """
    def __init__(self, vocab_size, embed_dim, num_heads, num_blocks, block_size, dropout_rate=0.2, **kwargs):
        super(NanoGPT, self).__init__(**kwargs)
        self.token_embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim, name="token_embedding")
        self.position_embedding = Embedding(input_dim=block_size, output_dim=embed_dim, name="position_embedding")

        # stack of transformer blocks
        self.transformer_blocks = Sequential(
            [
                TransformerBlock(
                    num_heads=num_heads,
                    head_size=embed_dim // num_heads,
                    embed_dim=embed_dim,
                    dropout_rate=dropout_rate,
                    name=f"transformer_block_{i}",
                )
                for i in range(num_blocks)
            ],
            name="transformer_blocks",
        )

        self.layer_norm = LayerNormalization(epsilon=1e-6, name="final_layer_norm")
        self.logits_dense = Dense(vocab_size, name="logits_dense")

    def call(self, x):
        """
        Forward pass for the NanoGPT model.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, time).

        Returns:
            tf.Tensor: Logits of shape (batch, time, vocab_size).
        """
        batch_size, seq_length = tf.shape(x)[0], tf.shape(x)[1]

        token_embeddings = self.token_embedding(x)  # (batch, time, embed_dim)

        # Positional embeddings
        positions = tf.range(start=0, limit=seq_length, delta=1)
        position_embeddings = self.position_embedding(positions)  # (time, embed_dim)
        position_embeddings = tf.expand_dims(position_embeddings, 0)  # (1, time, embed_dim)

        # Combine token and positional embeddings
        x = token_embeddings + position_embeddings  # (batch, time, embed_dim)

        # Pass through transformer blocks
        x = self.transformer_blocks(x)  # (batch, time, embed_dim)

        # Final layer normalization
        x = self.layer_norm(x)  # (batch, time, embed_dim)

        # Compute logits for each token in the vocabulary
        logits = self.logits_dense(x)  # (batch, time, vocab_size)

        return logits

    def generate(self, input_ids, max_new_tokens):
        """
        Generate new tokens based on the input sequence.

        Args:
            input_ids (tf.Tensor): Tensor of shape (batch, time) containing input token IDs.
            max_new_tokens (int): Number of tokens to generate.

        Returns:
            tf.Tensor: Tensor containing the generated token IDs.
        """
        for _ in range(max_new_tokens):
            # truncate input to the block size
            input_ids_truncated = input_ids[:, -BLOCK_SIZE:]

            logits = self.call(input_ids_truncated)  # (batch, time, vocab_size)

            # extract logits for the last time step
            last_logits = logits[:, -1, :]  # (batch, vocab_size)

            # sample the next token 
            sampled_ids = tf.random.categorical(last_logits, num_samples=1)  # (batch, 1)
            sampled_ids = tf.squeeze(sampled_ids, axis=-1)  # (batch,)

            # append the sampled token to the input_ids
            input_ids = tf.concat([input_ids, tf.expand_dims(sampled_ids, axis=-1)], axis=1)

        return input_ids

# -------------------------------
# Model Compilation and Training
# -------------------------------

def train_model(model, optimizer, train_dataset, test_dataset, epochs=10, generate_text_callback=True):
    """
    Train the model on the provided datasets.

    Args:
        model (tf.keras.Model): The model to train.
        train_dataset (tf.data.Dataset): Training dataset.
        test_dataset (tf.data.Dataset): Testing dataset.
        epochs (int, optional): Number of training epochs.
        generate_text_callback (bool, optional): Whether to generate text during training.

    Returns:
        dict: A dictionary with training loss history.
    """

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        for idx, (x, y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x)
                loss = loss_fn(y, logits)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            epoch_loss += loss.numpy()
            num_batches += 1

            if idx % 10 == 0:
                print(f"Epoch {epoch+1}, Step {idx}, Loss: {loss.numpy()}")

        avg_loss = epoch_loss / num_batches
        train_loss_history.append(avg_loss)

        if generate_text_callback:
            generated_text = generate_text(model, "Once upon a time", vectorizer, max_new_tokens=100)
            print('GENERATED TEXT AT EPOCH', epoch + 1, ':', generated_text)

    return {"loss": train_loss_history}

# Instantiate the model
print("Initializing NanoGPT model...")
gpt_model = NanoGPT(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_blocks=NUM_BLOCKS,
    block_size=BLOCK_SIZE,
    dropout_rate=DROPOUT_RATE,
)

# Optimizer definition
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-8)

# Train the model
print("Starting training...")
history = train_model(gpt_model, optimizer, train_dataset, test_dataset, epochs=EPOCHS, generate_text_callback=True)

# -------------------------------
# Saving the Model
# -------------------------------

def save_model(model, save_path="nano_gpt_model"):
    """
    Save the trained model to a specified path.

    Args:
        model (tf.keras.Model): The trained model.
        save_path (str, optional): Directory path to save the model.

    Returns:
        None
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.save(save_path)
    print(f"Model saved to {save_path}")

# Save the trained model
# save_model(gpt_model)


# ------------------------------- TOKEN DECODING -------------------------------
def decode_tokens(token_ids, vectorizer):
    """
    Decode a sequence of token IDs back to string.

    Args:
        token_ids (tf.Tensor): Tensor containing token IDs.
        vectorizer (TextVectorization): Fitted TextVectorization layer.

    Returns:
        str: Decoded string.
    """
    vocab = vectorizer.get_vocabulary()
    return "".join([vocab[token] for token in token_ids.numpy()])

# Example generation
def generate_text(model, start_string, vectorizer, max_new_tokens=50):
    """
    Generate text using the trained model starting from `start_string`.

    Args:
        model (NanoGPT): The trained NanoGPT model.
        start_string (str): The initial string to start generation.
        vectorizer (TextVectorization): Fitted TextVectorization layer.
        max_new_tokens (int, optional): Number of tokens to generate.

    Returns:
        str: Generated text string.
    """
    # Encode the start string
    input_ids = vectorizer([start_string])[0]
    input_ids = tf.expand_dims(input_ids, 0)  # (1, time)

    generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)

    # decode the generated tokens
    return decode_tokens(generated_ids[0], vectorizer)

