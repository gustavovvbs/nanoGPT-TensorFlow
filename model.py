import tensorflow as tf 
import numpy as np 
import os 

n_embed = 512
n_heads = 16
block_size = 8 
head_size = n_embed // n_heads 
vocab_size = 100
n_blocks = 6

with open('data.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

vocab = sorted(set(text))
vocab_size = len(vocab)
n_tokens = len(text)

text_vec_layer = tf.keras.layers.TextVectorization(max_tokens = vocab_size)
text_vec_layer.adapt(text)
encoded_text = text_vec_layer([text])[0]

n = int(len(text_vec_layer.get_vocabulary())*0.9)
raw_train_set = encoded_text[:n]
raw_test_set = encoded_text[n:]

def to_dataset(sequence, length, shuffle = False, seed = None, batch_size = 32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift = 1, drop_remainder = True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
    if shuffle:
        ds = ds.shuffle(100000, seed = seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)

train_set = to_dataset(raw_train_set, shuffle = True, length = 8)
test_set = to_dataset(raw_test_set, length = 8)


class AttentionHead(tf.keras.Model):
    def __init__(self, head_size, n_embed):
        super().__init__()
        self.key = tf.keras.layers.Dense(head_size)
        self.query = tf.keras.layers.Dense(head_size)
        self.value = tf.keras.layers.Dense(head_size)
        self.proj = tf.keras.layers.Dense(n_embed)

    def call(self, x):
        B, T, C = x.shape 

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        trig = tf.linalg.band_part(tf.ones(shape = (T, T)), -1, 0)

        att_w = (q @ tf.transpose(k, perm = [0, 2, 1])) * (head_size)**-0.5

        att_w = tf.where(trig == 0, -np.inf, att_w)

        att_w = tf.nn.softmax(att_w, axis = -1)

        output = att_w @ v

        return output 


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, head_size, n_heads, n_embed):
        super().__init__()
        self.heads = [AttentionHead(head_size, n_embed) for _ in range(n_heads)]
        self.proj = tf.keras.layers.Dense(n_embed)
        self.dropout = tf.keras.layers.Dropout(0.2)
        
    def call(self, x):
        atts = tf.concat([h(x) for h in self.heads], axis = -1)
        output = self.dropout(self.proj(atts))

        return output

class FullyC(tf.keras.Model):
    def __init__(self, n_embed):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(n_embed * 4)
        self.dense2 = tf.keras.layers.Dense(n_embed)
        self.dropout = tf.keras.layers.Dropout(0.2)
    def call(self, x):
        x = self.dense1(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.dense2(x)
        output = self.dropout(x)

        return output 

class Block(tf.keras.Model):
    def __init__(self, n_heads, n_embed):
        super().__init__()
        self.head_size = n_embed // n_heads 
        self.sa = MultiHeadAttention(self.head_size, n_heads, n_embed)
        self.fc = FullyC(n_embed)
        self.lnorm1 = tf.keras.layers.LayerNormalization()
        self.lnorm2 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = x + self.sa(self.lnorm1(x))
        x = x + self.fc(self.lnorm2(x))

        return x

class NanoGPT(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.embedding_matrix = tf.keras.layers.Embedding(vocab_size, n_embed)
        self.pos_emb = tf.keras.layers.Embedding(block_size, n_embed)
        
        self.blocks = tf.keras.Sequential([
            Block(n_heads, n_embed) for _ in range(n_blocks)
        ])

        self.lnorm = tf.keras.layers.LayerNormalization()
        self.flog = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        B, T = x.shape 

        token_embeddings = self.embedding_matrix(x) #(B, T, E)
        pos_embeddings = self.pos_emb(tf.range(T))#(T, E)
            
        x = token_embeddings + pos_embeddings #(B, T, E)
        x = self.blocks(x)
        x = self.lnorm(x)

        vocab_logits = self.flog(x) #(B, T, VOCAB_SIZE)

        return vocab_logits

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            croped_idx = idx[:, -block_size:]

            logits = self.call(croped_idx)

            last_ts = logits[:, -1, :] #(B, C)

            logits_norm = tf.nn.softmax(last_ts, axis = -1)

            sampled_idx = tf.random.categorical(logits_norm, num_samples=1, dtype=tf.int32) #(B, 1)

            idx = tf.concat([idx, sampled_idx], axis = 1)

        return idx



if __name__ == '__main__':
    gpt = NanoGPT()
    optimizer = tf.keras.optimizers.AdamW()
    ce = tf.keras.losses.SparseCategoricalCrossentropy()
    epochs = 3
    for i in range(epochs):
        with tf.GradientTape() as tape:
            for item in train_set:
                gen_logits = gpt(item[0])
                loss = ce(item[1], gen_logits)
                print(f'********EPOCH: {i}/{epochs}')
                print(f'Loss: {loss}')
            gradients = tape.gradient(loss, gpt.trainable_variables)
            optimizer.apply_gradients(zip(gradients, gpt.trainable_variables))




    

