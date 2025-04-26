import numpy as np

def load_data(file_path):
    data = open(file_path, 'r').read()
    chars = list(set(data))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    return data, chars, vocab_size, char_to_ix, ix_to_char
