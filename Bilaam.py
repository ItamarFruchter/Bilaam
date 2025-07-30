import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###############################################
#                     HP                      #
###############################################
# Learning hyper-parameters:
batch_size = 64
block_size = 256
max_iter = 5000
learning_rate = 3e-4
eval_interval = 200
eval_iters = 10
n_embed = 384
n_head = 6
num_of_blocks = 6
dropout = 0.2

###################################################
#                     Classes                     #
###################################################
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, eval_iters=eval_iters):
    out = {}
    
    model.eval()
    for split in ["train", "validation"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out


class Head(nn.Module):
    """
    One Self-Attention head.
    """
    def __init__(self, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False, device=device)
        self.query = nn.Linear(n_embed, head_size, bias=False, device=device)
        self.value = nn.Linear(n_embed, head_size, bias=False, device=device)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, device=device))) # Because tril isn't a parameter, pytorch wants us to register it like so.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x) # each one (B, T, C)
        v = self.value(x) # Also pass x through a linear layer. This is the "information" in this token.

        wei = q @ k.transpose(-2, -1) * (C ** -0.5) # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # [:T, :T] because x won't allways be of length 8 (in inference for example).
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHead(nn.Module):
    """
    Multiple Self-Attention head.
    """
    def __init__(self, num_heads, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed, device=device)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed, device=device),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed, device=device),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, n_embed, num_heads, dropout):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MultiHead(num_heads, head_size, dropout) # "Communication"
        self.ln1 = nn.LayerNorm(n_embed, device=device)
        self.ffwd = FeedForward(n_embed, dropout) # "Computation"
        self.ln2 = nn.LayerNorm(n_embed, device=device)

    def forward(self, x):
        x = self.sa(self.ln1(x)) + x # "+x" -> the residual connection.
        x = self.ffwd(self.ln2(x)) + x # Same
        return x


class Bilaam(nn.Module):
    """
    Biblical Language Model
    """
    def __init__(self, vocab_size, n_embed, num_heads, dropout, device):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed, device=device) # Adding intermidate layer (of size n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed, device=device)
        self.blocks = nn.Sequential(
            *[Block(n_embed, num_heads, dropout) for _ in range(num_of_blocks)]
        )
        self.ln_f = nn.LayerNorm(n_embed, device=device) # Final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size, device=device)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (Batch, Time, Channels (n_embed))
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # pos_emb is broadcasted for every part of the batch (its the same).
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C) # Reshape for the cross-entropy function
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is currently (B, T) array of indices
        for _ in range(max_new_tokens):
            # Crop to fit in context length:
            idx_cond = idx[:, -block_size:]
            # Get predictions:
            logits, loss = self(idx_cond) # (B,T,C), (1)
            # last timestep:
            logits = logits[:, -1, :] # Now (B,C)
            # Apply softmax:
            probs = F.softmax(logits, dim=1)
            # Sample:
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # Append to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx

#######################################################
#                       Running                       #
#######################################################
if __name__ == "main":
    print(" OLD VERSION - NO TOKENIZER ")
    # Processing Data:
    bible_data = open("./HebData/Hebrew_Bible-k_all_clean_eot.txt", "r", encoding="utf-8").read().splitlines()
    bible_data_clean = []
    print("Reading the bible:")
    for line in bible_data:
        ix = line.find(" ")
        if ix != -1:
            line = line[ix + 1:] # Remove chapter and verse numbers
        elif line == "<|endoftext|>" or line.strip() == "":
            continue
        else:
            print('\t', line.strip()) # Should only be book names

        if line.find("\xa0") != -1:
            line = line.replace("\xa0", " ")
        bible_data_clean.append(line.strip())
    bible = "\n".join(bible_data_clean) 
    chars = sorted(list(set(bible)))
    vocab_size = len(chars)
    print("Done!")
    print(f"Dataset length: {len(bible)} characters")

    stoi = {ch:i for i, ch in enumerate(chars)}
    itos = {i:ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(bible), dtype=torch.long, device=device)
    n = int(len(data) * 0.9)
    train_data = data[:n]
    val_data = data[n:]


    # Training
    m = Bilaam(vocab_size=vocab_size, n_embed=n_embed, num_heads=n_head, dropout=dropout, device=device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    def train(m, steps, eval_interval):
        lossi = {"train": [], "validation": []}
        optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
        for step in range(steps):
            xb, yb = get_batch('train')

            logits, loss = m(xb, yb)

            if step % eval_interval == 0:
                for k, v in estimate_loss(m).items():
                    lossi[k].append(v)
                

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        plt.plot(lossi['train'], label='Training Loss')
        plt.plot(lossi['validation'], label='Validation Loss')
        plt.legend(loc='upper right')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.show()
        print(f"Final loss:\n\tTrain - {lossi['train'][-1]}\n\tValidation - {lossi['validation'][-1]}")
        return lossi


    def gen_bible(m, pred_len, context=None):
        if context:
            idx = torch.tensor(encode(context))
        else:
            idx = torch.zeros((1,1), dtype=torch.long, device=device)

        preds = m.generate(idx, pred_len) # B predictions (but B=1)
        return decode(preds[0].tolist())


    print("Training...")
    train(m, steps=max_iter, eval_interval=eval_interval)
    # train(m, max_iter, eval_interval)
    print("Done! Generating:")
    print(gen_bible(m, 100))