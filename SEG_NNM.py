import torch
import json
import argparse
import re
from BiLSTM_CRF import BiLSTM_CRF
from torch.utils.data import DataLoader
from collections import namedtuple
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextDataset(Dataset):
    def __init__(self, text_path, chr_vocab_path):
        with open(text_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.chr_vocab = json.load(open(chr_vocab_path, 'r', encoding = 'utf-8'))
        self.sentences = self.split_sentences(self.text)

    def split_sentences(self, text):
        sentences = []
        for match in re.finditer(r'[^。？！\n]*[。？！\n]', text):
            sentence = match.group()
            if sentence.strip():
                sentences.append(sentence)
        return sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        char_ids = [self.chr_vocab[char] for char in sentence if char in self.chr_vocab]
        return torch.tensor(char_ids)

def collate_fn(data):
    words = [item for item in data]
    max_seq_len = max([t.shape[0] for t in words])
    batch_size = len(words)
    word_ids = torch.zeros(batch_size, max_seq_len).long()
    mask = torch.zeros(batch_size, max_seq_len).long()
    for idx, word in enumerate(words):
        word_ids[idx, :word.shape[0]] = word
        mask[idx, :word.shape[0]] = 1
    return word_ids, mask.bool()

def segment_text(data_loader, output_path, model, chr_vocab, tag_vocab):
    model.eval()
    with open(output_path, 'w', encoding='utf-8') as f:
        with torch.no_grad():
            for word_input, mask in data_loader:
                word_input = word_input.to(device)
                mask = mask.to(device)
                tag_seq = model.decode(word_input, mask)
                words = [chr_vocab[idx.item()] for idx in word_input[0]]
                tags = [tag_vocab[idx.item()] for idx in tag_seq[0]]
                sentence = ''.join([word if tag == 'M' or tag == 'B' 
                                    else word + ' ' for word, tag in zip(words, tags)])
                f.write(sentence + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='save_model/best.pt')
    parser.add_argument('--text', default='data/text.txt')
    parser.add_argument('--chr_vocab', default='data/chr_vocab.json')
    parser.add_argument('--tag_vocab', default='data/tag_vocab.json')
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    arg = parser.parse_args()

    dataset = TextDataset(arg.text, arg.chr_vocab)
    data_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    chr_vocab = json.load(open(arg.chr_vocab, 'r', encoding = 'utf-8'))
    chr_vocab = {v:k for k,v in chr_vocab.items()}
    tag_vocab = json.load(open(arg.tag_vocab, 'r', encoding = 'utf-8'))
    tag_vocab = {v:k for k,v in tag_vocab.items()}
    config = {
        'vocab_size': len(chr_vocab),
        'hidden_dim': arg.hidden_dim,
        'dropout': arg.dropout,
        'embedding_dim': arg.hidden_dim,
        'tag_size': len(tag_vocab),
        'gpu':True
    }
    config = namedtuple('config', config.keys())(**config)
    model = BiLSTM_CRF(config).to(device)

    # 加载最优模型的权重
    model.load_state_dict(torch.load(arg.save_path))
    segment_text(data_loader, 'seg_nnm.txt', model, chr_vocab, tag_vocab)