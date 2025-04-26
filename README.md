
# Character-Level RNN from Scratch

A simple, modular implementation of a character-level Recurrent Neural Network (RNN) for text generation, inspired by Karpathy’s [char-rnn](https://github.com/karpathy/char-rnn), built using only NumPy.

## Features

- Vanilla RNN implemented from scratch (no deep learning frameworks)
- Trains on any plain text file, learns to generate text character-by-character
- Modular code: data loading, model, and training are separated for clarity
- Easily extensible for experiments or learning

## Project Structure

```
char_rnn_project/
│
├── data/
│   └── input.txt             # Your training text file
│
├── src/
│   ├── data_utils.py         # Data loading and preprocessing
│   ├── model.py              # RNN model and sampling functions
│   └── train.py              # Training loop
│
├── outputs/
│   └── checkpoints/          # (optional) Saved model parameters
│
├── README.md
└── requirements.txt
```

## Usage

### 1. Prepare Data

Place your training text file as `data/input.txt`. You can use any large text file (e.g., Shakespeare, song lyrics, etc.).

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
cd src
python train.py
```

- The model will print generated text samples and the loss every 100 iterations.
- Training runs indefinitely; stop it manually when satisfied.

### 4. Customization

- Adjust hyperparameters (`hidden_size`, `seq_length`, `learning_rate`) in `train.py` or `model.py`.
- To use a different dataset, replace `data/input.txt` with your own text.

## File Descriptions

- `data_utils.py`: Loads data, creates vocabulary, and mappings.
- `model.py`: Defines the CharRNN class (forward, backward, sampling).
- `train.py`: Runs the training loop and prints samples.
- `requirements.txt`: Project dependencies (NumPy).

##  Example Output

```
----
To be, or not to be: that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
----
iter 100, loss: 47.123456
```

##  Extending

- Add model saving/loading (e.g., with `np.savez`).
- Implement LSTM/GRU cells for better results.
- Add temperature sampling for more/less creative output.
- Use batching for faster training.



