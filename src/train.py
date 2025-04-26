import numpy as np
from data_utils import load_data
from model import CharRNN

# Load data
data, chars, vocab_size, char_to_ix, ix_to_char = load_data('../data/input.txt')

# Hyperparameters
hidden_size = 100
seq_length = 25
learning_rate = 1e-1

# Initialize model
rnn = CharRNN(vocab_size, hidden_size, seq_length, learning_rate)

# Memory variables for Adagrad
mWxh, mWhh, mWhy = np.zeros_like(rnn.Wxh), np.zeros_like(rnn.Whh), np.zeros_like(rnn.Why)
mbh, mby = np.zeros_like(rnn.bh), np.zeros_like(rnn.by)

smooth_loss = -np.log(1.0/vocab_size) * seq_length
n, p = 0, 0
hprev = np.zeros((hidden_size, 1))

while True:
    if p+seq_length+1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1))
        p = 0
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    if n % 100 == 0:
        sample_ix = rnn.sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt, ))

    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = rnn.lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0:
        print('iter %d, loss: %f' % (n, smooth_loss))

    for param, dparam, mem in zip([rnn.Wxh, rnn.Whh, rnn.Why, rnn.bh, rnn.by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    p += seq_length
    n += 1
