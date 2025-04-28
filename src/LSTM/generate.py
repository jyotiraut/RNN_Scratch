import torch
from model import CharLSTM
from data_utils import load_data

def generate_sample(model, char_to_ix, ix_to_char, prime_str='A', length=1000, temp=0.8):
    model.eval()
    chars = [ch for ch in prime_str]
    inputs = torch.LongTensor([char_to_ix[ch] for ch in chars])
    hidden = model.init_hidden(1)
    
    for _ in range(length):
        output, hidden = model(inputs[-1].unsqueeze(0).unsqueeze(0).float(), hidden)
        probs = torch.softmax(output / temp, dim=-1).squeeze()
        next_char = torch.multinomial(probs, 1).item()
        chars.append(ix_to_char[next_char])
    
    return ''.join(chars)

# Load model and generate
data, _, _, char_to_ix, ix_to_char = load_data('../data/input.txt')
model = CharLSTM(len(char_to_ix), hidden_size=512, num_layers=2)
model.load_state_dict(torch.load('char_lstm.pth'))

print(generate_sample(model, char_to_ix, ix_to_char, prime_str="Once upon a time"))
