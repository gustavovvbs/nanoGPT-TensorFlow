import tensorflow as tf 
import numpy as np 
from model import * 
import time
import json

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

def load_config(config_file):
    with open(config_file, "r") as file:
        config = json.load(file)
    return config


class Solver():

    def __init__(self):
        config_args = load_config('config.json')
        print(config_args)
        self.VOCAB_SIZE = config_args["VOCAB_SIZE"]
        self.EMBED_DIM = config_args["EMBED_DIM"]
        self.BLOCK_SIZE = config_args["BLOCK_SIZE"]
        self.NUM_LAYERS = config_args["NUM_LAYERS"]
        self.DROPOUT_RATE = config_args["DROPOUT_RATE"]
        self.LEARNING_RATE = config_args["LEARNING_RATE"]
        self.BATCH_SIZE = config_args["BATCH_SIZE"]
        self.EPOCHS = config_args["EPOCHS"]
        self.TRAIN_SPLIT = config_args["TRAIN_SPLIT"]
        self.NUM_HEADS = config_args["NUM_HEADS"]
        self.HEAD_DIM = self.EMBED_DIM // self.NUM_HEADS

        self.model = NanoGPT(self.VOCAB_SIZE, self.EMBED_DIM, self.NUM_HEADS, self.NUM_LAYERS, self.BLOCK_SIZE, self.DROPOUT_RATE)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.LEARNING_RATE)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

    def train(self, train_dataset, val_dataset):
        for epoch in range(self.EPOCHS):
            for batch, (x, y) in enumerate(train_dataset):
                start = time.time()
                B, T = x.shape
                with tf.GradientTape() as tape:
                    logits = self.model(x, training = True)
                    loss = self.loss(y, logits)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                norm = tf.linalg.global_norm(gradients)
                print(f'Epoch: {epoch}, Step: {batch}, Loss: {loss}, Norm: {norm}, Tokens/s: {B*T/(time.time()-start)}')
        
            generated_text = generate_text(self.model, 'Once upon a time', 100)
            print(f'Generated text at the end of epoch {epoch}: {generated_text}')



if __name__ == '__main__':
    solver = Solver()
    raw_text = load_text_data('data.txt')
    encoded_text, vocab_size, vectorizer = preprocess_text(raw_text, max_tokens = 100)

    split_index = int(len(encoded_text) * solver.TRAIN_SPLIT)
    train_sequence = encoded_text[:split_index]
    test_sequence = encoded_text[split_index:]

    train_dataset = create_dataset(train_sequence, solver.BLOCK_SIZE, solver.BATCH_SIZE, shuffle = True, seed = 42)
    val_dataset = create_dataset(test_sequence, solver.BLOCK_SIZE, solver.BATCH_SIZE, shuffle = False)

    print(f'Training tokens: {len(train_sequence)}')
    solver.train(train_dataset, val_dataset)
