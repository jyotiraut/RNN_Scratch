import torch
from torch import nn, optim
from model import CharLSTM
from data_utils import load_data

# Hyperparameters
seq_length = 100
hidden_size = 512
num_layers = 2
learning_rate = 0.001
epochs = 5000

# Load data
data, chars, vocab_size, char_to_ix, ix_to_char = load_data('../data/input.txt')

# Convert characters to indices
data_indices = [char_to_ix[ch] for ch in data]

# Initialize model
model = CharLSTM(vocab_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    # Prepare batch
    start_idx = torch.randint(0, len(data_indices) - seq_length, (1,)).item()
    inputs = torch.LongTensor(data_indices[start_idx:start_idx+seq_length])
    targets = torch.LongTensor(data_indices[start_idx+1:start_idx+seq_length+1])
    
    # Forward pass
    hidden = model.init_hidden(1)
    model.zero_grad()
    output, hidden = model(inputs.unsqueeze(0).float(), hidden)
    loss = criterion(output.squeeze(0), targets)
    
    # Backward pass
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    
    # Print progress
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
        # Generate sample
        with torch.no_grad():
            sample = generate_sample(model, char_to_ix, ix_to_char, prime_str="The ")
            print(f'Generated text:\n{sample}\n')

# Save model
torch.save(model.state_dict(), 'char_lstm.pth')
