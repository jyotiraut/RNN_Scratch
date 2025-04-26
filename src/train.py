import numpy as np
from data_utils import load_data
from model import CharRNN

# Load training data
data, _, _, char_to_ix, ix_to_char = load_data('../data/input.txt')

# Initialize model
rnn = CharRNN(vocab_size=len(char_to_ix))

# Adagrad memory variables
mWxh, mWhh, mWhy = np.zeros_like(rnn.Wxh), np.zeros_like(rnn.Whh), np.zeros_like(rnn.Why)
mbh, mby = np.zeros_like(rnn.bh), np.zeros_like(rnn.by)

# Training state
smooth_loss = -np.log(1.0/len(char_to_ix)) * rnn.seq_length
n, p = 0, 0  # Iteration counter, data pointer
hprev = np.zeros((rnn.hidden_size, 1))

try:
    while True:
        # Reset hidden state if at end of data
        if p + rnn.seq_length + 1 >= len(data) or n == 0:
            hprev = np.zeros((rnn.hidden_size, 1))
            p = 0

        # Prepare inputs and targets
        inputs = [char_to_ix[ch] for ch in data[p:p+rnn.seq_length]]
        targets = [char_to_ix[ch] for ch in data[p+1:p+rnn.seq_length+1]]

        # Sample and print progress
        if n % 100 == 0:
            sample_ix = rnn.sample(hprev, inputs[0], 200)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print(f'\n---- Sample After Iteration {n} ----\n{txt}\n')

        # Forward-backward pass
        loss, dWxh, dWhh, dWhy, dbh, dby, hprev = rnn.lossFun(inputs, targets, hprev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        # Print loss
        if n % 100 == 0:
            print(f'Iteration {n}, Loss: {smooth_loss:.4f}')

        # Adagrad parameter update
        for param, dparam, mem in zip([rnn.Wxh, rnn.Whh, rnn.Why, rnn.bh, rnn.by],
                                    [dWxh, dWhh, dWhy, dbh, dby],
                                    [mWxh, mWhh, mWhy, mbh, mby]):
            mem += dparam * dparam
            param += -rnn.learning_rate * dparam / np.sqrt(mem + 1e-8)

        # Move data pointer
        p += rnn.seq_length
        n += 1

# In your KeyboardInterrupt exception block:
except KeyboardInterrupt:
    print("\nTraining stopped. Saving model...")
    np.savez('final_model.npz',
             Wxh=rnn.Wxh,  # Must match these exact names
             Whh=rnn.Whh,
             Why=rnn.Why,
             bh=rnn.bh,
             by=rnn.by,
             char_to_ix=char_to_ix,
             ix_to_char=ix_to_char)


