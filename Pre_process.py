import re
import json

def clean_date(file_path):
    with open(file_path, 'r', encoding='gbk') as f:
        text_lines = f.readlines()

    cleaned_sentences = []
    for line in text_lines:
        cleaned_sentence = re.sub(r'^\d{8}-\d{2}-\d{3}-\d{3}/m', '', line)
        if cleaned_sentence != '' and not cleaned_sentence.isspace():
            cleaned_sentences.append(cleaned_sentence.strip())

    with open('data/cleaned_date.txt', 'w', encoding='utf-8') as f:
        for sentence in cleaned_sentences:
            f.write(sentence + '\n')

def generate_dict_from_segmented_text(file_path, dict_path):
    word_dict = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            word_dict.update(words)
    
    with open(dict_path, 'w', encoding='utf-8') as f:
        json.dump(list(word_dict), f, ensure_ascii=False)

clean_date('data/ChineseCorpus199801.txt')

# 读取文件中的文本
with open('data/cleaned_date.txt', 'r', encoding='utf-8') as file:
    corpus = file.read()

# 分割每行
lines = corpus.strip().split('\n')

# 对每行进行处理
cleaned_lines = []
for line in lines:
    # 分割每个词
    words = line.split()
    # 对每个词进行处理
    cleaned_words = []
    for word in words:
        # 找到"/"的位置
        slash_index = word.find('/')
        if slash_index != -1:
            # 取出"/"之前的部分
            cleaned_word = word[:slash_index]
            cleaned_word = cleaned_word.replace('[', '')
            cleaned_words.append(cleaned_word)
    cleaned_lines.append(' '.join(cleaned_words))

# 将清洗后的词写入新的文件
with open('data/cleaned_words.txt', 'w', encoding='utf-8') as file:
    file.write('\n'.join(cleaned_lines))

generate_dict_from_segmented_text('data/cleaned_words.txt', 'data/word_dict.json')