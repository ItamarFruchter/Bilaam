import matplotlib.pyplot as plt
import regex as re

MAX_UTF_VAL = 255


class BilaamTokenizer:
    def __init__(self):
        super().__init__()
        self.merges = {}

    def _get_common_pair(self, ids):
        pairs = []
        for word in ids:
            for i,j in zip(word, word[1:]):
                pairs.append((i,j))
        count_d = dict.fromkeys(pairs, 0)

        for p in pairs:
            count_d[p] += 1
        
        top_pair = max(count_d.items(), key=lambda p: p[1])
        return top_pair

        
    def _merge_bytes(self, text, byte_pair, new_token):
        new_text = []
        i = 0
        for word in text:
            new_word = []
            i = 0
            while i < len(word):
                if i != len(word)-1 and word[i] == byte_pair[0] and word[i+1] == byte_pair[1]:
                    new_word.append(new_token)
                    i += 1
                else:
                    new_word.append(word[i])
                i += 1
            
            new_text.append(new_word)
        
        return new_text


    def BPE_encode_and_train(self, text, iters):
        gpt2pat_for_heb = re.compile(r""" ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        split_text = re.findall(gpt2pat_for_heb, text)
        split_bytes = [t.encode("utf-8") for t in split_text]
        split_ids = [[int(t) for t in s] for s in split_bytes]

        cur_text = split_ids
        merges = {}

        for i in range(iters):
            common_pair, count = self._get_common_pair(cur_text)
            new_token = MAX_UTF_VAL + i + 1
            cur_text = self._merge_bytes(cur_text, common_pair, new_token)
            merges[common_pair] = new_token

        self.merges = merges

        # Flatten the text:
        output = [c for word in cur_text for c in word]
        return output
    

    def BPE_encode(self, text):
        gpt2pat_for_heb = re.compile(r""" ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        split_text = re.findall(gpt2pat_for_heb, text)
        split_bytes = [t.encode("utf-8") for t in split_text]
        split_ids = [[int(t) for t in s] for s in split_bytes]

        cur_text = split_ids
        for pair, token in self.merges.items():
            cur_text = self._merge_bytes(cur_text, pair, token)
        
        output = [c for word in cur_text for c in word]
        return output


    def BPE_decode(self, ids):
        vocab = {idx: bytes([idx]) for idx in range(256)} # Default vocab
        for (c0, c1), idx in self.merges.items():
            vocab[idx] = vocab[c0] + vocab[c1] # Extending it. Concatanation.
        
        tokens = b"".join(vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
    

    def plot_compression(original_text, encoded_text):
        # Thank you, co-pilot:
        plt.figure(figsize=(12, 6))

        # Calculate lengths
        original_lengths = list(map(len, original_text))
        bpe_lengths = list(map(len, encoded_text))

        # Create histograms with better styling
        plt.hist(original_lengths, bins=range(1, max(original_lengths) + 2), 
                alpha=0.7, color='orange', label='Before BPE', edgecolor='black', linewidth=0.5)
        plt.hist(bpe_lengths, bins=range(1, max(bpe_lengths) + 2), 
                alpha=0.7, color='blue', label='After BPE', edgecolor='black', linewidth=0.5)

        # Add statistics
        original_avg = sum(original_lengths) / len(original_lengths)
        bpe_avg = sum(bpe_lengths) / len(bpe_lengths)
        compression_ratio = sum(original_lengths) / sum(bpe_lengths)

        plt.axvline(original_avg, color='darkorange', linestyle='--', linewidth=2, 
                label=f'Original Avg: {original_avg:.1f}')
        plt.axvline(bpe_avg, color='darkblue', linestyle='--', linewidth=2, 
                label=f'BPE Avg: {bpe_avg:.1f}')

        plt.title(f'Token Length Distribution: Before vs After BPE\nCompression Ratio: {compression_ratio:.2f}x', 
                fontsize=14, fontweight='bold')
        plt.xlabel('Token Length (bytes)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # Add text box with statistics (positioned to avoid legend overlap)
        stats_text = f'Total tokens: {len(original_text)} → {len(encoded_text)}\n'
        stats_text += f'Total bytes: {sum(original_lengths)} → {sum(bpe_lengths)}\n'
        stats_text += f'Avg length: {original_avg:.1f} → {bpe_avg:.1f}'

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=9)

        plt.tight_layout()
        plt.show()