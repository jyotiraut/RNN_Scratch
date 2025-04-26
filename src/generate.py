import numpy as np
from model import CharRNN

def generate_text(model_path, num_chars=500, seed_text="\n"):
    # Load saved model CORRECTLY
    checkpoint = np.load(model_path, allow_pickle=True)
    
    # Verify keys exist (debugging step)
    print("Checkpoint contains:", checkpoint.files)
    
    # Load parameters
    Wxh = checkpoint['Wxh']
    Whh = checkpoint['Whh']
    Why = checkpoint['Why']
    bh = checkpoint['bh']
    by = checkpoint['by']
    char_to_ix = checkpoint['char_to_ix'].item()  # .item() for dictionaries
    ix_to_char = checkpoint['ix_to_char'].item()

    # Initialize model
    rnn = CharRNN(len(char_to_ix))
    rnn.Wxh = Wxh
    rnn.Whh = Whh
    rnn.Why = Why
    rnn.bh = bh
    rnn.by = by

    # Process seed text
    h = np.zeros((rnn.hidden_size, 1))
    seed_ix = [char_to_ix[ch] for ch in seed_text]
    
    if len(seed_ix) == 0:
        seed_ix = [np.random.randint(len(char_to_ix))]
    
    # Warm up the RNN with seed text
    for ix in seed_ix[:-1]:
        x = np.zeros((rnn.vocab_size, 1))
        x[ix] = 1
        h = np.tanh(np.dot(rnn.Wxh, x) + np.dot(rnn.Whh, h) + rnn.bh)
    
    # Start generation
    generated = list(seed_ix)
    x = np.zeros((rnn.vocab_size, 1))
    x[seed_ix[-1]] = 1
    
    for _ in range(num_chars):
        h = np.tanh(np.dot(rnn.Wxh, x) + np.dot(rnn.Whh, h) + rnn.bh)
        y = np.dot(rnn.Why, h) + rnn.by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(rnn.vocab_size), p=p.ravel())
        x = np.zeros((rnn.vocab_size, 1))
        x[ix] = 1
        generated.append(ix)
    
    # Convert to text
    return ''.join(ix_to_char[ix] for ix in generated)

if __name__ == "__main__":
    generated = generate_text('final_model.npz', num_chars=1000, seed_text="Once upon a time  ")
    print("\nGenerated Text:\n" + generated)
