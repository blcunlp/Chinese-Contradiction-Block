from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import sys

import tensorflow as tf

# from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin

import logging

from collections import defaultdict
from qa_model import Encoder
from qa_model import Decoder
from qa_model import QASystem

logging.basicConfig(level=logging.INFO)

#allow you to define hyperparams of model
tf.app.flags.DEFINE_float("learning_rate", 0.00001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.2, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 50, "Number of epochs to train.")
# reduce # of units in hidden layer (default is 200)
tf.app.flags.DEFINE_integer("state_size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("question_size",50, "The output size of your question.")#length of question
tf.app.flags.DEFINE_integer("output_size", 50, "The output size of your model.")#length of passage
tf.app.flags.DEFINE_integer("embedding_size", 300, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "./data/", "SQuAD directory (default ./data/squad)")
#tf.app.flags.DEFINE_string("train_dir", "./data/data_train/results/20190615_034548", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("train_dir", "data/data_train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adamax", "adam / sgd / adamax")
tf.app.flags.DEFINE_integer("print_every", 100, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/vocab.txt", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")

FLAGS = tf.app.flags.FLAGS #goes everywhere else in model


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        # index to word
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        # word to index
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir

def load_dataset(f1, f2, f3, batch_size):
    fd1, fd2, fd3 = open(f1), open(f2), open(f3)

    question_batch = []
    paragraph_batch = []
    answer_batch = []

    while True:
        line1, line2, line3 = (fd1.readline().rstrip(),
                                fd2.readline().rstrip(),
                                fd3.readline().rstrip())
        if not line1:
            break
        ## TODO: need to trim > max length in test set
        # TODO: stop removing longer training paragraphs
        try:
          paragraph = [int(x) for x in  line2.split()]
          if len(paragraph) >= FLAGS.output_size:
              continue
          question = [int(x) for x in  line1.split()]
          if len(question) >= FLAGS.question_size:
              continue
          answer = [int(x) for x in line3.split()]
        except:
          pass
        else: 
          question_batch.append(question)
          paragraph_batch.append(paragraph)
          answer_batch.append(answer)

        if len(question_batch) == batch_size:
            yield  (question_batch, paragraph_batch, answer_batch)
            question_batch = []
            paragraph_batch = []
            answer_batch = []

def generate_histograms(dataset):
    question_lengths = []
    paragraph_lengths = []
    answer_lengths = []

    for q, p, a in dataset:
        for i in range(len(q)):
            question = len(q[i]) 
            paragraph = len(p[i]) 
            #span = (a[i][1] - a[i][0]) 
            question_lengths.append(question)
            paragraph_lengths.append(paragraph)
            #answer_lengths[span] += 1

    print(max(question_lengths))
    print(max(paragraph_lengths))

def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    # use .readlines() to load file ourselves
    # use python generator
    question_path = pjoin(FLAGS.data_dir, "data_train/train_tokenH.txt")
    paragraph_path = pjoin(FLAGS.data_dir, "data_train/train_tokenP.txt")
    answer_path = pjoin(FLAGS.data_dir, "data_train/train_index.txt")

    val_question_path = pjoin(FLAGS.data_dir, "data_test/test_tokenH.txt")
    val_paragraph_path = pjoin(FLAGS.data_dir, "data_test/test_tokenP.txt")
    val_answer_path = pjoin(FLAGS.data_dir, "data_test/test_index.txt")

    # loads embedding
    FLAGS.embed_path = FLAGS.embed_path or pjoin("data", "sgns.merge.char.npz")
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.txt")
    vocab, rev_vocab = initialize_vocab(vocab_path) # one is list and one is dict
    # for testing
    # dataset = [(1,1,1), (1,1,1)]
    dataset = load_dataset(question_path, paragraph_path, answer_path, FLAGS.batch_size)
    val_dataset = load_dataset(val_question_path, val_paragraph_path, val_answer_path, FLAGS.batch_size)
    #generate_histograms(dataset)
    #generate_histograms(val_dataset)


    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(size=FLAGS.state_size, output_size=FLAGS.output_size)

    qa = QASystem(encoder, decoder, FLAGS)

    # log file
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # start training
    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        qa.train(sess, dataset, val_dataset, save_train_dir, rev_vocab)

        #qa.evaluate_answer(sess, dataset, vocab, FLAGS.evaluate, log=True)

if __name__ == "__main__":
    tf.app.run()
