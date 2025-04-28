
# Character-Level RNN from Scratch

A simple, modular implementation of a character-level Recurrent Neural Network (RNN) for text generation, inspired by Karpathy’s [char-rnn](https://github.com/karpathy/char-rnn), built using only NumPy.

## Features

Pure NumPy implementation (no deep learning frameworks required)

Learns to generate text character-by-character from any input text

Supports seed text for context-aware text generation

Adagrad optimizer for stable training

Modular code: data loading, model, training, and generation scripts





```
char_rnn_project/
├── data/
│   └── input.txt          # Your training text
├── src/
│   ├── data_utils.py      # Data loading
│   ├── model.py           # RNN model
│   ├── train.py           # Training script
│   └── generate.py        # Text generation script
├── README.md
└── requirements.txt


## Usage

### 1. Prepare Data

- Place your training text file as `data/input.txt`.
- Use any large plain text file (e.g., Shakespeare, song lyrics, etc.).

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Train the Model

```bash
cd src
python train.py
```
- The model will print generated samples and loss every 100 iterations.
- Training runs until you stop it with `Ctrl+C`.
- The model is automatically saved as `final_model.npz` when you stop training.

---

### 4. Generate Text with a Seed

```bash
python generate.py
```
- By default, generates 1000 characters starting with the seed `"The "`.
- To change the seed or number of characters, edit the last lines of `generate.py`:
  ```python
  generated = generate_text('final_model.npz', num_chars=1000, seed_text="Once upon a time, ")
  print("\nGenerated Text:\n" + generated)
  ```

#### How Seed Text Works

- The seed text is used to "prime" the RNN, setting its hidden state as if it had already read those characters.
- The generated text will continue from the context of your seed.

---

## File Descriptions

- **data_utils.py**: Loads text and creates character-index mappings.
- **model.py**: Defines the RNN model, forward/backward pass, and sampling.
- **train.py**: Runs the training loop, prints samples, and saves the model.
- **generate.py**: Loads the trained model and generates text, starting from your chosen seed.





