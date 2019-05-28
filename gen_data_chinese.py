#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
from collections import namedtuple
from typing import List, Dict

SentenceTriplet = namedtuple('SentenceTriplet', 'sent, head, tail, relation')


def load_data(data_file: Path, tag_file: Path) -> List[SentenceTriplet]:
    print("读取数据：句子-关系三元组...")
    data: List[SentenceTriplet] = []
    multi_tag = 0
    with data_file.open() as sent, tag_file.open() as tag:
        for data_line, tag_line in zip(sent, tag):
            sent_id, head, tail, sent = data_line.strip('\n').split('\t')
            tag_sent_id, tag = tag_line.strip('\n').split('\t')
            assert(sent_id == tag_sent_id)
            if tag.find(' ') != -1:
                multi_tag += 1
            else:
                data.append(SentenceTriplet(sent, head, tail, int(tag)))
    print(f'丢弃{multi_tag}个多标签数据，暂不处理')
    return data


def load_word_vector(file_name: Path):
    print("读取词向量...")
    word2id = {}
    word_vec_mat = []
    word_count = 0
    vec_dim = 0
    with file_name.open() as f:
        header = f.readline()
        word_count, vec_dim = map(int, header.strip('\n').split(' '))
        for line in f:
            word, *vec = line.strip(' \n').split(' ')
            vec = [float(n) for n in vec]
            word2id[word] = len(word2id)
            word_vec_mat.append(vec)
    assert(word_count == len(word2id))
    assert(vec_dim == len(word_vec_mat[0]))
    print(f"读到 {word_count} 个词， 向量维度 {vec_dim}")

    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)
    word_vec_mat.append(np.random.normal(loc=0, scale=0.05, size=vec_dim))
    word_vec_mat.append(np.zeros(vec_dim, dtype=np.float32))
    word_vec_mat = np.array(word_vec_mat, dtype=np.float32)
    print("词向量读取完毕")
    return word2id, word_vec_mat


def vectorize_data(original_data, sentence_length, word2id: Dict[str, int], is_training=True):
    sen_total = len(original_data)
    sen_word = np.zeros((sen_total, sentence_length), dtype=np.int64)
    sen_pos1 = np.zeros((sen_total, sentence_length), dtype=np.int64)
    sen_pos2 = np.zeros((sen_total, sentence_length), dtype=np.int64)
    sen_mask = np.zeros((sen_total, sentence_length, 3), dtype=np.float32)
    sen_len = np.zeros((sen_total), dtype=np.int64)
    for i, record in enumerate(original_data):
        if i % 1000 == 0:
            print(i)
        words = record.sent.split()
        # sen_len
        sen_len[i] = min(len(words), sentence_length)
        # sen_word
        for j, word in enumerate(words):
            if j < sentence_length:
                sen_word[i][j] = word2id.get(word, word2id['UNK'])

        for j in range(j + 1, sentence_length):
            sen_word[i][j] = word2id['BLANK']

        pos1, pos2 = words.index(record.head), words.index(record.tail)
        pos_min = min(pos1, pos2)
        pos_max = max(pos1, pos2)
        for j in range(sentence_length):
            # sen_pos1, sen_pos2
            sen_pos1[i][j] = j - pos1 + sentence_length
            sen_pos2[i][j] = j - pos2 + sentence_length
            # sen_mask
            if j >= sen_len[i]:
                sen_mask[i][j] = [0, 0, 0]
            elif j - pos_min <= 0:
                sen_mask[i][j] = [100, 0, 0]
            elif j - pos_max <= 0:
                sen_mask[i][j] = [0, 100, 0]
            else:
                sen_mask[i][j] = [0, 0, 100]
    print("保存数据集矩阵")
    if is_training:
        name_prefix = "train"
    else:
        name_prefix = "test"
    np.save(out_path / f'{name_prefix}_word.npy', sen_word)
    np.save(out_path / f'{name_prefix}_pos1.npy', sen_pos1)
    np.save(out_path / f'{name_prefix}_pos2.npy', sen_pos2)
    np.save(out_path / f'{name_prefix}_mask.npy', sen_mask)


in_path = Path("./chinese_data/open_data/")
out_path = Path("./chinese_data/")

# 读词向量
word2id, word_vec_mat = load_word_vector(Path("./chinese_data/sgns.weibo.word"))

print("=====训练数据=====")
original_data = load_data(in_path/'sent_train.txt', in_path/'sent_relation_train.txt')
vectorize_data(original_data, 120, word2id, is_training=True)

print("=====开发测试数据=====")
original_data = load_data(in_path/'sent_dev.txt', in_path/'sent_relation_dev.txt')
vectorize_data(original_data, 120, word2id, is_training=False)

print('保存词向量')
np.save(out_path / 'vec.npy', word_vec_mat)