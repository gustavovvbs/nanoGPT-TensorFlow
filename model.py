import tensorflow as tf 
import numpy as np 
import os 

n_embed = 512
n_heads = 16
block_size = 8 
head_size = n_embed // n_heads 












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

        trig = tf.linalg.band_part(tf.ones(shape = (T, T)), -1, 0) * (head_size)**-0.5

        att_w = q @ tf.transpose(k, perm = [0, 2, 1])

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

blockex = Block(n_heads, n_embed)
outputex = blockex(tf.random.normal(shape = (16, 8, n_embed)))

print(outputex)