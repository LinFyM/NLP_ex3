import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import json
from CRF import CRF
import argparse
from collections import namedtuple
import matplotlib.pyplot as plt
from tqdm import tqdm

# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTM_CRF(nn.Module):
    def __init__(self, config):
        super(BiLSTM_CRF, self).__init__()
        self.target_size = config.tag_size
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout
        self.embedding_dim = config.embedding_dim
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.drop = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_dim * 2, self.target_size + 2)
        self.crf = CRF(self.target_size, config.gpu)

    def get_tags(self, word_input):
        embedded = self.embedding(word_input)
        hidden = None
        lstm_feats, hidden = self.lstm(embedded, hidden)
        feats = self.drop(lstm_feats)
        feats = self.hidden2tag(feats)
        return feats

    def forward(self, word_input, mask, tags):
        feats = self.get_tags(word_input)
        loss = self.crf(feats, mask, tags)
        return loss

    def decode(self, word_input, mask):
        feats = self.get_tags(word_input)
        tag_seq = self.crf.decode(feats, mask)
        return tag_seq

class SEG_dataset(Dataset):
    def __init__(self, data_path, chr_vocab_path, tag_vocab_path):
        self.data = self.load_data(data_path)
        self.chr_vocab = json.load(open(chr_vocab_path, 'r', encoding = 'utf-8'))
        self.tag_vocab = json.load(open(tag_vocab_path, 'r', encoding = 'utf-8'))
        print(len(self.data))

    def load_data(self, data_path):
        data = []
        with open(data_path, 'r', encoding = 'utf-8') as f:
            sentence = []
            tags = []
            for line in f:
                if line == '\n':
                    data.append([sentence, tags])
                    sentence = []
                    tags = []
                else:
                    word, tag = line.strip().split(' ')
                    sentence.append(word)
                    tags.append(tag)
        if sentence:
            data.append([sentence, tags])
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sentence, tags = self.data[index]
        char_ids = [self.chr_vocab[char] for char in sentence]
        tag_ids = [self.tag_vocab[tag] for tag in tags]
        return torch.tensor(char_ids), torch.tensor(tag_ids)

def collate_fn(data):
    words = [item[0] for item in data]
    tags = [item[1] for item in data]
    max_seq_len = max([t.shape[0] for t in words])
    batch_size = len(words)
    word_ids = torch.zeros(batch_size, max_seq_len).long()
    mask = torch.zeros(batch_size, max_seq_len).long()
    tag_ids = torch.zeros(batch_size, max_seq_len).long()
    for idx, (word, tag) in enumerate(zip(words, tags)):
        word_ids[idx, :word.shape[0]] = word
        tag_ids[idx, :tag.shape[0]] = tag
        mask[idx, :word.shape[0]] = 1
    return word_ids, mask.bool(), tag_ids

def train(model, config, train_dataset, val_dataset, tag_vocab):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    best_acc = 0

    val_accuracies = []

    for epoch in range(config.num_epoch):
        for data in tqdm(train_dataset, desc=f"Training epoch {epoch+1}/{config.num_epoch}"):
            word_input, mask, tags = [d.to(device) for d in data]
            loss = model(word_input, mask, tags)
            loss.backward()
            optimizer.step()
            model.zero_grad()

        acc = evaluate(model, val_dataset, tag_vocab)
        print('epoch: {}, loss: {:.4f}, Acc Score:{}'.format(epoch, loss.item(), acc))
        if acc > best_acc:
            best_acc = acc

            torch.save(model.state_dict(), config.save_path)
            
        val_accuracies.append(acc)

    return val_accuracies

def evaluate(model, dataset, tag_vocab):
    predictions, true_labels = sequence_tag(model, dataset, tag_vocab)
    accuracy = cal_acc(predictions, true_labels)
    return accuracy

def sequence_tag(model, dataset, tag_vocab):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for i, (word_input, mask, tags) in enumerate(dataset):
            word_input, mask, tags = word_input.to(device), mask.to(device), tags.to(device)
            tag_seq = model.decode(word_input, mask)
            predicted_tags = [[tag_vocab[idx.item()] for idx in seq] for seq in tag_seq]

            predictions.extend(predicted_tags)

            true_tags = [[tag_vocab[idx.item()] for idx in seq] for seq in tags]
            true_labels.extend(true_tags)

            # # 如果是第一句，打印预测的标签和真实的标签的前5个元素
            # if i == 0:
            #     print("Original Predicted Tags for the first sentence:", tag_seq[0][:5])
            #     print("Mapped Predicted Tags for the first sentence:", predicted_tags[0][:5])
            #     print("Original True Tags for the first sentence:", tags[0][:5])
            #     print("Mapped True Tags for the first sentence:", true_tags[0][:5])

    model.train()

    return predictions, true_labels

def cal_acc(pred_seq, golden_seq):
    correct, total = 0, 0
    for pred, gold in zip(pred_seq, golden_seq):
        for p, g in zip(pred, gold):
            if p == g:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0

def plot_accuracies(val_accuracies, test_accuracy):
    epochs = range(1, len(val_accuracies) + 1)

    plt.plot(epochs, val_accuracies, 'b', label='Val acc')
    plt.scatter(len(epochs), test_accuracy, color='red', label='Test acc')  # 测试集准确率点
    plt.title('Val acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='save_model/best.pt')
    parser.add_argument('--train', default='data/train.txt')
    parser.add_argument('--test', default='data/test.txt')
    parser.add_argument('--val', default='data/val.txt')
    parser.add_argument('--chr_vocab', default='data/chr_vocab.json')
    parser.add_argument('--tag_vocab', default='data/tag_vocab.json')
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--lr', default=0.00005, type=float)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eval_interval', default=100, type=int)
    arg = parser.parse_args()

    train_dataset = SEG_dataset(arg.train, arg.chr_vocab, arg.tag_vocab)
    val_dataset = SEG_dataset(arg.val, arg.chr_vocab, arg.tag_vocab)
    test_dataset = SEG_dataset(arg.test, arg.chr_vocab, arg.tag_vocab)
    train_loader = DataLoader(train_dataset, batch_size=arg.batch_size, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=arg.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
    chr_vocab = json.load(open(arg.chr_vocab, 'r', encoding = 'utf-8'))
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
    val_accuracies = train(model, arg, train_loader, val_loader, tag_vocab)

    model.load_state_dict(torch.load(arg.save_path))
    test_accuracy = evaluate(model, test_loader, tag_vocab)
    print(f"Accuracy: {test_accuracy}")
    plot_accuracies(val_accuracies, test_accuracy)