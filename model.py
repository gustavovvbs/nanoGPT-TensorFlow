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
# Model Components
# -------------------------------

class MultiHeadAttention(Model):
    def __init__(self, num_heads, head_dim, dropout_rate, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.all_head_dim = num_heads * head_dim

        self.qkv = Dense(self.all_head_dim * 3, use_bias=False, name="qkv")
        self.out_proj = Dense(self.all_head_dim, use_bias=False, name="out_proj")
        self.dropout = Dropout(dropout_rate)

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
    def __init__(self, embed_dim, dropout_rate, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.dense1 = Dense(embed_dim * 4, activation='gelu', name="ff_dense1")
        self.dense2 = Dense(embed_dim, name="ff_dense2")
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training):
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

class TransformerBlock(Model):
    def __init__(self, num_heads, head_dim, embed_dim, dropout_rate, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.layer_norm1 = LayerNormalization(epsilon=1e-5, name="ln1")
        self.mha = MultiHeadAttention(num_heads=num_heads, head_dim=head_dim, dropout_rate=dropout_rate, name="multi_head_attention")
        self.layer_norm2 = LayerNormalization(epsilon=1e-5, name="ln2")
        self.ffn = FeedForward(embed_dim=embed_dim, dropout_rate=dropout_rate, name="feed_forward")

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
    def __init__(self, vocab_size, embed_dim, num_heads, num_blocks, block_size,dropout_rate, **kwargs):
        super(NanoGPT, self).__init__(**kwargs)
        self.token_embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim, name="token_embedding")
        self.position_embedding = Embedding(input_dim=block_size, output_dim=embed_dim, name="position_embedding")
        self.dropout = Dropout(dropout_rate)
        self.blocks = [TransformerBlock(num_heads=num_heads, head_dim=embed_dim // num_heads, embed_dim=embed_dim, dropout_rate=dropout_rate, name=f"block_{i}") for i in range(num_blocks)]
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