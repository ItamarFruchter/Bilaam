from Bilaam import Bilaam
from BilaamTokenizer import BilaamTokenizer
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class BilaamRunner:
    DEFAULT_HP = {
        "batch_size": 64,
        "block_size": 256,
        "max_iter": 5000,
        "learning_rate": 3e-4,
        "eval_interval": 200,
        "eval_iters": 10,
        "n_embed": 384,
        "n_head": 6,
        "num_of_blocks": 6,
        "dropout": 0.2,
        "tokenize_iters": 48,
    }

    def __init__(self, checkpoint_path = "./misc/bilaam.checkpoint", should_train=False, hyper_params=DEFAULT_HP, train_mock=False):
        self.model = None
        self.checkpoint_path = checkpoint_path
        self.train_data = None
        self.val_data = None
        self.tokenizer = BilaamTokenizer()

        # HPs:
        self.batch_size = hyper_params["batch_size"]
        self.block_size = hyper_params["block_size"]
        self.max_iter = hyper_params["max_iter"]
        self.learning_rate = hyper_params["learning_rate"]
        self.eval_interval = hyper_params["eval_interval"]
        self.eval_iters = hyper_params["eval_iters"]
        self.n_embed = hyper_params["n_embed"]
        self.n_head = hyper_params["n_head"]
        self.num_of_blocks = hyper_params["num_of_blocks"]
        self.dropout = hyper_params["dropout"]
        self.tokenize_iters = hyper_params["tokenize_iters"]

        if should_train:
            print(f"{'='*20} Loading the data... {'='*20}")
            self.load_data(mock=train_mock)
            print(f"{'='*20} Training... {'='*20}")
            self.train(self.max_iter)
        else:
            self.load_checkpoint()


    def load_data(self, mock=False):
        # Processing Data:
        bible_data = open("./misc/HebData/Hebrew_Bible-k_all_clean_eot.txt", "r", encoding="utf-8").read().splitlines()
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
        if mock:
            self.max_iter = 5
            bible = bible[:5000]
        print(f"Dataset length: {len(bible)} characters")

        # Tokenize and encode:
        print("Encoding and tokenizing...")
        encoded_bible = self.tokenizer.BPE_encode_and_train(bible, self.tokenize_iters)

        chars = sorted(list(set(bible))) # Just for reference
        
        # Create mapping from sparse BPE tokens to dense range [0, vocab_size-1]
        unique_tokens = sorted(list(set(encoded_bible)))
        self.vocab_size = len(unique_tokens)
        
        # Create bidirectional mapping
        self.sparse_to_dense = {sparse_id: dense_id for dense_id, sparse_id in enumerate(unique_tokens)}
        self.dense_to_sparse = {dense_id: sparse_id for sparse_id, dense_id in self.sparse_to_dense.items()}
        
        print(f"Vocab size: {self.vocab_size} (dense range 0-{self.vocab_size-1})")
        print(f"Original sparse token range: {min(unique_tokens)}-{max(unique_tokens)}")
        print(f"Original character vocab size: {len(chars)}")
        
        # Map encoded_bible from sparse to dense
        encoded_bible_dense = [self.sparse_to_dense[token] for token in encoded_bible]

        # Create tensor only once using dense encoding
        data = torch.tensor(encoded_bible_dense, dtype=torch.long, device=device)
        n = int(len(data) * 0.9)
        self.train_data = data[:n]
        self.val_data = data[n:]
        print("Done!")


    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
    

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        
        self.model.eval()
        for split in ["train", "validation"]:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split)
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()

        return out


    def train(self, steps, plot=True):
        if self.model is None:
            self.model = Bilaam(vocab_size=self.vocab_size, n_embed=self.n_embed, num_heads=self.n_head, dropout=self.dropout, device=device)

        lossi = {"train": [], "validation": []}
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        for step in tqdm(range(steps)):
            xb, yb = self.get_batch('train')

            logits, loss = self.model(xb, yb)

            if step % self.eval_interval == 0:
                for k, v in self.estimate_loss().items():
                    lossi[k].append(v)
                

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if plot:
            plt.plot(lossi['train'], label='Training Loss')
            plt.plot(lossi['validation'], label='Validation Loss')
            plt.legend(loc='upper right')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title('Training vs Validation Loss')
            plt.show()
        print(f"Final loss:\n\tTrain - {lossi['train'][-1]}\n\tValidation - {lossi['validation'][-1]}")
        self.lossi = lossi
        return lossi


    def generate(self, pred_len, context=None):
        if context:
            # Encode context and map to dense
            sparse_tokens = self.tokenizer.BPE_encode(context)
            dense_tokens = [self.sparse_to_dense[token] for token in sparse_tokens]
            idx = torch.tensor([dense_tokens], dtype=torch.long, device=device)
        else:
            idx = torch.zeros((1,1), dtype=torch.long, device=device)

        # Generate using dense tokens
        dense_preds = self.model.generate(idx, pred_len)
        
        # Map back to sparse for decoding
        sparse_preds = [self.dense_to_sparse[token.item()] for token in dense_preds[0]]
        return self.tokenizer.BPE_decode(sparse_preds)
    

    def save_model(self):
        checkpoint = {
            'model': self.model,
            'tokenizer': self.tokenizer,
            'dense2sparse': self.dense_to_sparse,
            'sparse2dense': self.sparse_to_dense,
        }
        torch.save(checkpoint, self.checkpoint_path)


    def load_checkpoint(self):
        try:
            checkpoint = torch.load(self.checkpoint_path, weights_only=False)
        except FileNotFoundError:
            print(f"Checkpoint not found at {self.checkpoint_path}")
        self.model = checkpoint['model']
        self.tokenizer = checkpoint['tokenizer']
        self.dense_to_sparse = checkpoint['dense2sparse']
        self.sparse_to_dense = checkpoint['sparse2dense']