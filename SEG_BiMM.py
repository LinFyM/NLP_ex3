import json

def FMM(word_dict, text, max_length):
    # 正向最大匹配
    result = []
    index = 0
    while index < len(text):
        for size in range(min(max_length, len(text) - index), 0, -1):
            piece = text[index:index+size]
            if piece in word_dict or len(piece) == 1:
                result.append(piece)
                index += size
                break
        else:  # 如果没有找到匹配的词，将当前字符添加到结果中
            result.append(text[index])
            index += 1
    return result

def BMM(word_dict, text, max_length):
    # 反向最大匹配
    result = []
    index = len(text)
    while index > 0:
        for size in range(min(max_length, index), 0, -1):
            piece = text[index-size:index]
            if piece in word_dict or len(piece) == 1:
                result.append(piece)
                index -= size
                break
        else:  # 如果没有找到匹配的词，将当前字符添加到结果中
            result.append(text[index])
            index += 1
    return result[::-1]

def BiMM(word_dict, text, max_length):
    # 双向最大匹配
    FMM_result = FMM(word_dict, text, max_length)
    BMM_result = BMM(word_dict, text, max_length)
    if len(FMM_result) < len(BMM_result):
        return FMM_result
    elif len(FMM_result) > len(BMM_result):
        return BMM_result
    else:
        return FMM_result if FMM_result < BMM_result else BMM_result
    
def load_dict(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        word_dict = set(json.load(f))
    max_length = max(len(word) for word in word_dict)
    return word_dict, max_length

def segment_text(file_path, dict_path, output_path):
    word_dict, max_length = load_dict(dict_path)
    with open(file_path, 'r', encoding='utf-8') as f, open(output_path, 'w', encoding='utf-8') as out_f:
        for line in f:
            text = line.strip()
            result = BiMM(word_dict, text, max_length)
            out_f.write(' '.join(result) + '\n')

segment_text('data/text.txt', 'data/word_dict.json', 'seg_bimm.txt')