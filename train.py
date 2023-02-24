import torch
import torch.nn.functional as F
from transformer_efficient import miniGPT, block_size, device

## Train Params ##
current_file = "texts.txt"
load_path = "./saved_models/multi_batch.pt"
save_path = "./saved_models/multi_batch.pt"
num_heads = 4
num_blocks = 2
batch_size = 64
num_itrs = 10000
learning_rate = 4e-3
##################

def get_data():
    with open(current_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))

    print(chars)

    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:]

def get_vocab():
    f = open(current_file, 'r', encoding='utf-8')
    f.seek(0)
    text = f.read()
    f.close()
    chars = sorted(list(set(text)))
    return chars

def get_batch(data, batch_size):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def forward(model, x, vocab_size=None, targets=None):
    cross_entropy = torch.nn.CrossEntropyLoss()
    logits = model(x)
    loss = None

    if targets != None:
        loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

    return logits, loss

def train(load_path=None, save_path=None):
    t, v = get_data() # training and val data
    vocab = get_vocab()

    model = miniGPT(num_heads, num_blocks, len(vocab)).to(device=device) # want num_heads to be a divisor of n_embd
    
    if load_path != None:
        model.load_state_dict(torch.load(load_path))
    
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for i in range(num_itrs):
        optimizer.zero_grad()
        xb, yb = get_batch(t, batch_size)
        logits, loss = forward(model, xb, len(vocab), yb)

        if i % 1000 == 0:
            print(loss.item())
            
        loss.backward()        
        optimizer.step()

    if save_path != None:
        torch.save(model.state_dict(), save_path)

@torch.no_grad()
def eval(model_path, set="val"):
    vocab = get_vocab()
    model = miniGPT(num_heads, num_blocks, len(vocab))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    t, v = get_data()

    # Move model and data to device
    model = model.to(device)

    if set == "val":
        xval, yval = get_batch(v, v.shape[0])
        val_logits, val_loss = forward(model, xval, len(vocab), yval)
        print("The validation loss is", val_loss.item())
    else:
        xtr, ytr = get_batch(t, t.shape[0])
        tr_logits, tr_loss = forward(model, xtr, len(vocab), ytr)
        print("The train loss is", tr_loss.item())

@torch.no_grad()
def infer(model_path, prior=""):
    vocab = get_vocab()
    model = miniGPT(num_heads, num_blocks, len(vocab)).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    stoi = { ch:i for i,ch in enumerate(vocab) }    
    context = torch.zeros((1, block_size), dtype=torch.long).to(device) # 1, 8
    if prior != "":
        for i, c in enumerate(prior):
            context[0, i] = stoi[c]

    itos = { i:ch for i,ch in enumerate(vocab) }

    output = prior

    print("Model Loaded.")

    for i in range(500):
        logits, loss = forward(model, context)
        logits = logits.softmax(dim=-1).squeeze() # 1, block_size, vocab_size

        # sample from distribution using torch.multinomial
        samples = torch.multinomial(logits, 1) # 1, block_size, 1
        pred = samples[-1]

        output += itos[pred.item()]
        context = torch.cat((context[:, 1:], pred.view(1, -1)), dim=1)
    
    print(output)

# train(load_path, save_path)
eval(load_path, "val")
infer(load_path)