import json
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split

def word2tags(words):
    tags = []
    for word in words:
        if len(word) == 1:
            tags.append((word, 'S'))
        else:
            word_tags = [(word[0], 'B')] + [(word[i], 'M') for i in range(1, len(word) - 1)] + [(word[-1], 'E')]
            tags.extend(word_tags)
    return tags

# 读取文件中的文本
with open('data/cleaned_words.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# 使用句号、问号、感叹号或换行符来分割句子，并保留分隔符
sentences = []
for match in re.finditer(r'[^。？！\n]*[。？！\n]', text):
    sentence = match.group()
    if sentence.strip():  # 删除空句子
        sentences.append(sentence)

tagged_sentences = []
tag_vocab = defaultdict(int)
chr_vocab = defaultdict(int)
for sentence in sentences:
    words = sentence.split()
    tags = word2tags(words)
    tagged_sentences.append(tags)
    for word, tag in tags:
        tag_vocab[tag] += 1
        for chr in word:
            chr_vocab[chr] += 1

# 将数据分割为训练集、验证集和测试集
train_sentences, other_sentences = train_test_split(tagged_sentences, test_size=0.2, random_state=42)
val_sentences, test_sentences = train_test_split(other_sentences, test_size=0.5, random_state=42)  # 0.5 x 0.2 = 0.1

# 将训练集、验证集和测试集保存到文件
with open('data/train.txt', 'w', encoding='utf-8') as file:
    for sentence in train_sentences:
        for word, tag in sentence:
            file.write(f"{word} {tag}\n")
        file.write("\n")

with open('data/val.txt', 'w', encoding='utf-8') as file:
    for sentence in val_sentences:
        for word, tag in sentence:
            file.write(f"{word} {tag}\n")
        file.write("\n")

with open('data/test.txt', 'w', encoding='utf-8') as file:
    for sentence in test_sentences:
        for word, tag in sentence:
            file.write(f"{word} {tag}\n")
        file.write("\n")

# 将tag_vocab和chr_vocab保存到json文件
with open('data/tag_vocab.json', 'w', encoding='utf-8') as file:
    json.dump({tag: i for i, tag in enumerate(tag_vocab.keys())}, file, ensure_ascii=False)

with open('data/chr_vocab.json', 'w', encoding='utf-8') as file:
    json.dump({chr: i for i, chr in enumerate(chr_vocab.keys())}, file, ensure_ascii=False)