# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import argparse

from six.moves import urllib

from tensorflow.python.platform import gfile
from tqdm import *
import numpy as np
from os.path import join as pjoin

_PAD = b"<pad>"
_SOS = b"<sos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _UNK]

PAD_ID = 0
SOS_ID = 1
UNK_ID = 2

def setup_args():
    parser = argparse.ArgumentParser()
    home = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    vocab_dir = os.path.join("c_data", "")
    glove_dir = os.path.join("Embedding", "")
    source_dir = os.path.join("c_data","")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--vocab_dir", default=vocab_dir)
    parser.add_argument("--glove_dim", default=300, type=int)
    parser.add_argument("--random_init", default=True, type=bool)
    return parser.parse_args()


def basic_tokenizer(sentence):
     return sentence.strip().split()


def initialize_vocabulary(vocabulary_path):
    # map vocab to word embeddings
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def process_glove(args, vocab_list, save_path, size=4e5, random_init=True):
    """
    :param vocab_list: [vocab]
    :return:
    """
    if not gfile.Exists(save_path + ".npz"):
        #glove_path = os.path.join(args.glove_dir, "glove.6B.{}d.txt".format(args.glove_dim))
        glove_path = os.path.join(args.glove_dir, "sgns.merge.char")
        if random_init:
            glove = np.random.randn(len(vocab_list), args.glove_dim)
        else:
            glove = np.zeros((len(vocab_list), args.glove_dim))
        with open(glove_path, 'r') as fh:
            for line in tqdm(fh, total=size):
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                if word in vocab_list:
                    idx = vocab_list.index(word)
                    glove[idx, :] = vector
                #if word.capitalize() in vocab_list:
                #    idx = vocab_list.index(word.capitalize())
                #    glove[idx, :] = vector
                #if word.lower() in vocab_list:
                #    idx = vocab_list.index(word.lower())
                #    glove[idx, :] = vector
                #if word.upper() in vocab_list:
                #    idx = vocab_list.index(word.upper())
                #    glove[idx, :] = vector

        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))


def create_vocabulary(vocabulary_path, data_paths, tokenizer=None):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_paths)))
        vocab = {}
        for path in data_paths:
            #with open(path, mode="rb") as f:
            with open(path, mode="r") as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("processing line %d" % counter)
                    tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                    for w in tokens:
                        if w in vocab:
                            vocab[w] += 1
                        else:
                            vocab[w] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print("Vocabulary size: %d" % len(vocab_list))
        #with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
            for w in vocab_list:
                #vocab_file.write(w + b"\n")
                vocab_file.write(str(w)+"\n")


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None):
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with open(data_path, mode="r") as data_file:
            with open(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 5000 == 0:
                        print("tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer)
                    #with gfile.GFile(target_path, mode="w") as tokens_file:
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


if __name__ == '__main__':
    args = setup_args()
    vocab_path = pjoin(args.vocab_dir, "vocab.txt")

    train_path = pjoin(args.source_dir, "/data_trn")
    valid_path = pjoin(args.source_dir, "/data_dev")
    dev_path = pjoin(args.source_dir, "/data_dev")

    create_vocabulary(vocab_path,
                      [pjoin(args.source_dir, "data_train/train_p.txt"),
                       pjoin(args.source_dir, "data_train/train_h.txt"),
                       pjoin(args.source_dir, "data_dev/dev_p.txt"),
                       pjoin(args.source_dir, "data_dev/dev_h.txt")])
    vocab, rev_vocab = initialize_vocabulary(pjoin(args.vocab_dir, "vocab.txt"))

    # ======== Trim Distributed Word Representation =======
    # If you use other word representations, you should change the code below

    #process_glove(args, rev_vocab, args.source_dir + "/glove.trimmed.{}".format(args.glove_dim),
    process_glove(args, rev_vocab, args.source_dir + "sgns.merge.char",
                  random_init=args.random_init)

    # ======== Creating Dataset =========
    # We created our data files seperately
    # If your model loads data differently (like in bulk)
    # You should change the below code

    #x_train_dis_path = train_path + "/train_tokenP.txt"
    x_train_dis_path = pjoin(args.source_dir,"data_train/train_tokenP.txt")
    y_train_ids_path = pjoin(args.source_dir,"data_train/train_tokenH.txt")
   
    data_to_token_ids(pjoin(args.source_dir, "data_train/train_p.txt"), x_train_dis_path, vocab_path)
    data_to_token_ids(pjoin(args.source_dir, "data_train/train_h.txt"), y_train_ids_path, vocab_path)

    #x_dis_path = valid_path + "/dev_tokenP.txt"
    #y_ids_path = valid_path + "/dev_tokenH.txt"
    x_dis_path = pjoin(args.source_dir,  "data_dev/dev_tokenP.txt")
    y_ids_path = pjoin(args.source_dir,"data_dev/dev_tokenH.txt")
    data_to_token_ids(pjoin(args.source_dir, "data_dev/dev_p.txt"), x_dis_path, vocab_path)
    data_to_token_ids(pjoin(args.source_dir, "data_dev/dev_h.txt"), y_ids_path, vocab_path)
