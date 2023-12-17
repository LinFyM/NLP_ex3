import re
from collections import Counter

def subword_tokenization(filename, max_compressions=100):
    with open(filename, 'r') as f:
        text = f.read()

    # 合并所有连续的空格为一个空格
    text = re.sub(' +', ' ', text)

    compressions = 0
    compression_dict = {}
    while compressions < max_compressions:
        bigrams = re.findall(r'(?<= )[^ ]+ [^ ]+(?= )|(?<=^)[^ ]+ [^ ]+(?= )|(?<= )[^ ]+ [^ ]+(?=$)', text)
        bigram_counts = Counter(bigrams)
        most_common_bigram = bigram_counts.most_common(1)
        if most_common_bigram[0][1] < 2:
            break
        most_common_bigram = most_common_bigram[0][0]
        compression_dict[chr(128 + compressions)] = most_common_bigram
        text = text.replace(most_common_bigram, chr(128 + compressions))
        compressions += 1

    # 在所有的空格前添加@@
    text = text.replace(' ', '@@')

    # 还原文本
    for i in range(compressions-1, -1, -1):
        text = text.replace(chr(128 + i), compression_dict[chr(128 + i)])

    # 去除所有的空格
    text = text.replace(' ', '')

    # 在所有@@后面加上一个空格
    text = text.replace('@@', '@@ ')

    return text

filename = 'seg_bimm.txt'  # 经过分词的文本文件
compressed_text = subword_tokenization(filename)

# 将子词压缩的结果保存到一个文件中
with open('subwords.txt', 'w') as f:
    f.write(compressed_text)