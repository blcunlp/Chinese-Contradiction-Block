from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import time
import logging
import random
import codecs
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from evaluate import exact_match_score, f1_score
from qa_data import PAD_ID
from adamax import AdamaxOptimizer

logging.basicConfig(level=logging.INFO)

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    elif opt == "adamax":
        optfn = AdamaxOptimizer
    else:
        assert (False)
    return optfn

class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim # wat is this
        # pseudocode

    def encode_preprocess(self, inputs, masks, keep_prob, scope="", reuse=False): #wat is masks
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        # symbolic function takes in Tensorflow object, returns tensorflow object
        # pseudocode
        self.cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.size, state_is_tuple=True), output_keep_prob=keep_prob)
        with tf.variable_scope(scope):
            #(out_fw, out_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.cell, self.cell, inputs, sequence_length=masks, dtype=tf.float64)
            (out_fw, out_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.cell, self.cell, inputs, sequence_length=masks, dtype=tf.float64)
        #return tf.concat((out_fw, out_bw), 2)
        return tf.concat(2,[out_fw, out_bw])

    def encode_match(self, input_q, input_p, masks_p, keep_prob, scope=""):
        self.match_cell = tf.nn.rnn_cell.DropoutWrapper(MatchLSTMCell(self.size, input_q, state_is_tuple=True), output_keep_prob=keep_prob)
        with tf.variable_scope(scope) as scope_match:
            (out_fw, out_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.match_cell, self.match_cell, input_p, \
                sequence_length=masks_p, dtype=tf.float64)
            scope_match.reuse_variables()
        return tf.concat(2,[out_fw, out_bw])

class MatchLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    def __init__(self, num_units, input_q, state_is_tuple, scope=None):
        self.H_q = input_q
        self.size_q = self.H_q.get_shape().as_list()[1]
        super(MatchLSTMCell, self).__init__(num_units, state_is_tuple=state_is_tuple)

    def __call__(self, inputs, state, scope=None):
        # Params:
        # inputs - h_p_t
        # state - h_r_{t-1}
        with tf.variable_scope("matchlstm") as scope_lstm:
            W_q = tf.get_variable("W_q", shape=(2 * self._num_units, self._num_units),
                        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            W_p = tf.get_variable("W_p", shape=(2 * self._num_units, self._num_units),
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            W_r = tf.get_variable("W_r", shape=(self._num_units, self._num_units),
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            b_p = tf.get_variable("b_p", shape=(self._num_units,),
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            w = tf.get_variable("w", shape=(self._num_units, 1),
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            b = tf.get_variable("b", shape=(),
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            H_q = tf.reshape(self.H_q, [-1, 2 * self._num_units])
            H_qW_q = tf.reshape(tf.matmul(H_q, W_q), [-1, self.size_q, self._num_units])
            tempsum = tf.matmul(inputs, W_p) + tf.matmul(state[1], W_r) + b_p
            G = tf.tanh(H_qW_q + tf.expand_dims(tempsum, axis=1)) # doing broadcasting, but can tf.expand_dims(tempsum, axis=1)), tf.tile([1, size_q, 1])
            Gw = tf.reshape(tf.matmul(tf.reshape(G, [-1, self._num_units]), w), [-1, self.size_q])
            alpha = tf.nn.softmax(Gw + b)
            alpha = tf.expand_dims(alpha, axis=1)
            H_qalpha = tf.reshape(tf.matmul(alpha, self.H_q), [-1, 2 * self._num_units])
            z = tf.concat(1,[inputs, H_qalpha])
        output, new_state = super(MatchLSTMCell, self).__call__(z, state, scope)
        scope_lstm.reuse_variables()
        return output, new_state


class Decoder(object):
    def __init__(self, size, output_size):
        self.size = size
        self.output_size = output_size

    def decode(self, inputs, masks, keep_prob, scope):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        # given: h_r 
        # TODO: CUT DOWN TO BATCH_SIZE
        # each 2-d TF variable
        self.cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.size, state_is_tuple=True), output_keep_prob=keep_prob)
        self.size_p = inputs.get_shape().as_list()[1]

        with tf.variable_scope(scope) as scope_b:
            V = tf.get_variable("V", shape=(2 * self.size, self.size),
                        initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            W_a = tf.get_variable("W_a", shape=(self.size, self.size),
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            b_a = tf.get_variable("b_a", shape=(1, self.size),
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            v = tf.get_variable("v", shape=(self.size, 1),
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            c = tf.get_variable("c", shape=(),
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)
            # calculations for start prediction
            H_r = tf.reshape(inputs, [-1, 2 * self.size])
            H_rV = tf.reshape(tf.matmul(H_r, V), [-1, self.size_p, self.size])
            # TODO: check that first hidden state is 0?
            # W_ah_a = tf.expand_dims(tf.matmul(h_a, W_a) + b_a, axis=1)
            W_ah_a = tf.expand_dims(b_a, axis=1)
            F_k = tf.reshape(tf.tanh(H_rV + W_ah_a), [-1, self.size])
            F_kv = tf.reshape(tf.matmul(F_k, v), [-1, self.size_p])
            b_s = F_kv + c # can expand dim of c if needed
            b_h = tf.expand_dims(tf.nn.softmax(b_s), axis=1)
            z = tf.reshape(tf.matmul(b_h, inputs), [-1, 2 * self.size])
            # get hidden state for start
            h_s, _ = tf.nn.dynamic_rnn(self.cell, tf.expand_dims(z, axis=1), dtype=tf.float64)
            # calculations for end prediction
            W_eh_e = tf.matmul(tf.reshape(h_s, [-1, self.size]), W_a) + b_a
            F_k_e = tf.reshape(tf.tanh(H_rV + tf.expand_dims(W_eh_e, axis=1)), [-1, self.size])
            F_kv_e = tf.reshape(tf.matmul(F_k_e, v), [-1, self.size_p])
            b_e = F_kv_e + c # not softmaxed so we can use softmax_cross_entropy in calculating loss
            # b_s not softmaxed, a_s softmaxed
            mask_vectors = tf.sequence_mask(masks, maxlen=self.output_size, dtype=tf.float64)
            #log_mask_vectors = tf.log(mask_vectors)
            #b_s = b_s + log_mask_vectors
            #b_e = b_e + log_mask_vectors
            b_s = b_s * mask_vectors
            b_e = b_e * mask_vectors
            scope_b.reuse_variables()
        return (b_s, b_e, tf.nn.softmax(b_s), tf.nn.softmax(b_e))


class QASystem(object):
    def __init__(self, encoder, decoder, FLAGS, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        # ==== constants ====

        # ==== set up placeholder tokens ========
        self.FLAGS = FLAGS
        self.paragraphs_placeholder = tf.placeholder(tf.int32, shape=[None, self.FLAGS.output_size])
        self.questions_placeholder = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_size])
        self.q_masks_placeholder = tf.placeholder(tf.int32, shape=[None])
        self.p_masks_placeholder = tf.placeholder(tf.int32, shape=[None])
        self.start_answer = tf.placeholder(tf.int32, shape=[None])
        self.end_answer = tf.placeholder(tf.int32, shape=[None])
        self.keep_placeholder = tf.placeholder(tf.float64, shape=None)
        self.start_answer_re = tf.placeholder(tf.int32, shape=[None])
        self.end_answer_re = tf.placeholder(tf.int32, shape=[None])

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0), reuse=False) as scope:
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()
            self.setup_training_op()
            scope.reuse_variables()

        # ==== set up training/updating procedure ====
        pass

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        encoder = Encoder(self.FLAGS.state_size, self.FLAGS.embedding_size)
        H_q = encoder.encode_preprocess(self.questions_var, self.q_masks_placeholder, self.keep_placeholder, scope="question")
        H_p = encoder.encode_preprocess(self.paragraphs_var, self.p_masks_placeholder, self.keep_placeholder, scope="paragraph")
        H_r = encoder.encode_match(H_q, H_p, self.p_masks_placeholder, self.keep_placeholder, scope="Hr")
        H_r_re = encoder.encode_match(H_p, H_q, self.q_masks_placeholder, self.keep_placeholder, scope="Hr_re")
        decoder = Decoder(self.FLAGS.state_size, self.FLAGS.output_size)
        self.b_s, self.b_e, self.a_s, self.a_e = decoder.decode(H_r, self.p_masks_placeholder, self.keep_placeholder, scope="decode")
        self.b_s_re, self.b_e_re, self.a_s_re, self.a_e_re = decoder.decode(H_r_re, self.q_masks_placeholder, self.keep_placeholder, scope="decode_re")

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with tf.variable_scope("loss"):
            loss_s = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.start_answer, logits=self.b_s)
            loss_e = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.end_answer, logits=self.b_e)
            loss_s_re = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.start_answer_re, logits=self.b_s_re)
            loss_e_re = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.end_answer_re, logits=self.b_e_re)
            self.loss = loss_s + loss_e+ loss_s_re + loss_e_re

    def setup_training_op(self):
        optimizer = get_optimizer(self.FLAGS.optimizer)(self.FLAGS.learning_rate)
        gradients, variables = map(list, zip(*optimizer.compute_gradients(self.loss)))
        self.grad_norm = tf.global_norm(gradients)
        gradients, _ = tf.clip_by_global_norm(gradients, self.FLAGS.max_gradient_norm)
        grads_and_vars = zip(gradients, variables)
        self.train_op = optimizer.apply_gradients(grads_and_vars)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with tf.variable_scope("embeddings"):
            # load data
            glove_matrix = np.load(self.FLAGS.embed_path)['glove']
            embeddings = tf.Variable(glove_matrix, trainable=False)
            self.questions_var = tf.nn.embedding_lookup(embeddings, self.questions_placeholder)
            self.paragraphs_var = tf.nn.embedding_lookup(embeddings, self.paragraphs_placeholder)

    def optimize(self, session, train_q, train_p, train_a, q_masks, p_masks):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x
        input_feed[self.questions_placeholder] = train_q
        input_feed[self.paragraphs_placeholder] = train_p
        input_feed[self.q_masks_placeholder] = q_masks
        input_feed[self.p_masks_placeholder] = p_masks
        input_feed[self.keep_placeholder] = 1.0 - self.FLAGS.dropout
        

        a_starts = np.array([a[0] for a in train_a])
        a_ends = np.array([a[1] for a in train_a])
        a_starts_re = np.array([a[2] for a in train_a])
        a_ends_re = np.array([a[3] for a in train_a])

        input_feed[self.start_answer] = a_starts
        input_feed[self.end_answer] = a_ends
        input_feed[self.start_answer_re] = a_starts_re
        input_feed[self.end_answer_re] = a_ends_re

        output_feed = [self.train_op, self.loss, self.grad_norm]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_q, valid_p, valid_a, q_masks, p_masks):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        input_feed[self.paragraphs_placeholder] = valid_p
        input_feed[self.questions_placeholder] = valid_q
        input_feed[self.p_masks_placeholder] = p_masks
        input_feed[self.q_masks_placeholder] = q_masks
        input_feed[self.keep_placeholder] = 1.0

        a_starts = np.array([a[0] for a in valid_a])
        a_ends = np.array([a[1] for a in valid_a])
        a_starts_re = np.array([a[2] for a in valid_a])
        a_ends_re = np.array([a[3] for a in valid_a])

        input_feed[self.start_answer] = a_starts
        input_feed[self.end_answer] = a_ends
        input_feed[self.start_answer_re] = a_starts_re
        input_feed[self.end_answer_re] = a_ends_re

        output_feed = [self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_q, test_p, q_masks, p_masks):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x
        input_feed[self.paragraphs_placeholder] = test_p
        input_feed[self.questions_placeholder] = test_q
        input_feed[self.p_masks_placeholder] = p_masks
        input_feed[self.q_masks_placeholder] = q_masks
        input_feed[self.keep_placeholder] = 1.0

        output_feed = [self.a_s, self.a_e, self.a_s_re, self.a_e_re]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_q, test_p, q_masks, p_masks):

        yp, yp2,yp3,yp4 = self.decode(session, test_q, test_p, q_masks, p_masks)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)
        a_s_re = np.argmax(yp3, axis=1)
        a_e_re = np.argmax(yp4, axis=1)

        return (a_s, a_e, a_s_re, a_e_re)

    def validate(self, sess, valid_dataset, log=False):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function (at end of each epoch)

        :return:
        """
        valid_cost = 0
        num_seen = 0
        for q, p, a in valid_dataset:
            # questions in valid_dataset may need to be clipped
            q_lengths = [min(len(x), self.FLAGS.question_size) for x in q]
            p_lengths = [min(len(x), self.FLAGS.output_size) for x in p]
            q = [question[:self.FLAGS.question_size] + [PAD_ID] * (self.FLAGS.question_size - min(len(question), self.FLAGS.question_size)) for question in q]
            p = [paragraph[:self.FLAGS.output_size] + [PAD_ID] * (self.FLAGS.output_size - min(len(paragraph), self.FLAGS.output_size)) for paragraph in p]
            out = self.test(sess, q, p, a, q_lengths, p_lengths)
            valid_cost += sum(out[0])
            num_seen += len(out[0])

        average_valid_cost = float(valid_cost) / float(num_seen)

        if log:
            logging.info("Validate cost: {}".format(average_valid_cost))

        return valid_cost

    """
    def evaluate_answer(self, session, dataset, dataset_name, rev_vocab, log=False):
        f=codecs.open("predict.txt","w") 
        acc_as = 0.0
        acc_ae = 0.0
        acc_all=0.0
        count=0
        #random_indices = random.sample(xrange(len(dataset)), len(dataset)-1)
        #random_indices = random.sample(xrange(len(dataset)), len(dataset))
        # a is list of tuples
        for q,p,a in dataset:
            length=len(q)
            if count >= length: break
            q_lengths = [min(len(x), self.FLAGS.question_size) for x in q]
            p_lengths = [min(len(x), self.FLAGS.output_size) for x in p]
            q = [question[:self.FLAGS.question_size] + [PAD_ID] * (self.FLAGS.question_size - min(len(question), self.FLAGS.question_size)) for question in q]
            p = [paragraph[:self.FLAGS.output_size] + [PAD_ID] * (self.FLAGS.output_size - min(len(paragraph), self.FLAGS.output_size)) for paragraph in p]
            a_s, a_e, a_s_re, a_e_re = self.answer(session, q, p, q_lengths, p_lengths)
            f.write(str(a_s)+"\t"+str(a_e)+"\t"+str(a_s_re)+"\t"+str(a_e_re)+"\n")
            

            predict_a=np.reshape(np.array([a_s,a_e]),(self.FLAGS.batch_size,2))
            predict_a_re=np.reshape(np.array([a_s_re,a_e_re]),(self.FLAGS.batch_size,2))

            true_a_s=[l_a[0] for l_a in a]
            true_a_e=[l_a[1] for l_a in a]
            true_a_s_re=[l_a[2] for l_a in a]
            true_a_e_re=[l_a[3] for l_a in a]

            true_a=np.reshape(np.array([true_a_s,true_a_e]),(self.FLAGS.batch_size,2))
            true_a_re=np.reshape(np.array([true_a_s_re,true_a_e_re]),(self.FLAGS.batch_size,2))

            predict_all=np.reshape(np.array([predict_a,predict_a_re]),(self.FLAGS.batch_size,4))
            true_all=np.reshape(np.array([true_a_s,true_a_e,true_a_s_re,true_a_e_re]),(self.FLAGS.batch_size,4))
            #true_all=np.reshape(np.array([a,a_re]),(self.FLAGS.batch_size,4))

            acc_as=np.mean(np.equal(a_s,true_a_s),dtype=np.float64)
            acc_ae=np.mean(np.equal(a_e,true_a_e),dtype=np.float64)
            acc_p=np.mean(np.equal(predict_a,true_a),dtype=np.float64)

            acc_as_re=np.mean(np.equal(a_s_re,true_a_s_re),dtype=np.float64)
            acc_ae_re=np.mean(np.equal(a_e_re,true_a_e_re),dtype=np.float64)
            acc_h=np.mean(np.equal(predict_a_re,true_a_re),dtype=np.float64)
            #acc_re=np.mean(np.equal(predict_a_re.reshape(self.FLAGS.batch_size,2),a_re))

            acc_all=np.mean(np.equal(predict_all,true_all))
        if log:
            logging.info("dataset_name%s - acc_as: %f, acc_ae: %f,acc_p: %f"%(dataset_name, acc_as,acc_ae,acc_p))
            logging.info("dataset_name%s - acc_as_re: %f, acc_ae_re: %f,acc_h: %f"%(dataset_name, acc_as_re,acc_ae_re,acc_h))
            logging.info("acc_all: %f"%(acc_all))
    """

    def evaluate_answer(self, session, dataset, dataset_name, rev_vocab, sample=100, log=False):
        f1_p = 0.0
        f1_q = 0.0
        em_p = 0.0
        em_q = 0.0

        count = 0
        sample=len(dataset)
        random_indices = random.sample(xrange(len(dataset)), sample)
        # a is list of tuples
        for index in random_indices:
            q, p, a = dataset[index]
            if count >= sample: break

            q_lengths = [min(len(x), self.FLAGS.question_size) for x in q]
            p_lengths = [min(len(x), self.FLAGS.output_size) for x in p]
            q = [question[:self.FLAGS.question_size] + [PAD_ID] * (self.FLAGS.question_size - min(len(question), self.FLAGS.question_size)) for question in q]
            p = [paragraph[:self.FLAGS.output_size] + [PAD_ID] * (self.FLAGS.output_size - min(len(paragraph), self.FLAGS.output_size)) for paragraph in p]
            a_s, a_e,a_s_re, a_e_re = self.answer(session, q, p, q_lengths, p_lengths)
            #print('predict_out:')
            #print('{}\n{}\n{}\n{}'.format(a_s,a_e,a_s_re,a_e_re))

            #print('pline:')
            for i in range(len(q)):
                answer = ' '.join(map(lambda x: rev_vocab[x], p[i][a_s[i]:a_e[i] + 1])) #first pred block
                true_answer = ' '.join(map(lambda x: rev_vocab[x], p[i][a[i][0]:a[i][1] + 1])) #first true block
                #print('{}\n'.format(p[i]))
                #print('{}\n{}\n'.format(answer,true_answer))
                #p[i]

                f1_p += f1_score(answer, true_answer)
                em_p += exact_match_score(answer, true_answer)
                count += 1
                if count >= sample: break

            #print('hline:')
            for i in range(len(p)):
                answer = ' '.join(map(lambda x: rev_vocab[x], q[i][a_s_re[i]:a_e_re[i] + 1]))
                true_answer = ' '.join(map(lambda x: rev_vocab[x], q[i][a[i][2]:a[i][3] + 1]))
                #print('{}\n'.format(q[i]))
                #print('{}\n{}\n'.format(answer,true_answer))

                f1_q += f1_score(answer, true_answer)
                em_q += exact_match_score(answer, true_answer)
                count += 1
                if count >= sample: break

        f1_p /= sample
        em_p /= sample
        f1_q /= sample
        em_q /= sample

        if log:
            logging.info("{} - F1_p: {}, EM_p: {}, for {} samples".format(dataset_name, f1_p, em_p, sample))
            logging.info("{} - F1_q: {}, EM_q: {}, for {} samples".format(dataset_name, f1_q, em_q, sample))

    def train(self, session, dataset, val_dataset, train_dir, rev_vocab):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        dataset = list(dataset)
        val_dataset = list(val_dataset)

        #logging.info("Evaluating initial")
        #val_loss = self.validate(session, val_dataset, log=True)
        #self.evaluate_answer(session, dataset, "Train", rev_vocab, log=True)
        #self.evaluate_answer(session, val_dataset, "Validation", rev_vocab, log=True)

        # split into train and test loops?
        num_processed = 0
        for e in range(self.FLAGS.epochs):
            random.shuffle(dataset)
            for q, p, a in dataset:
                tic = time.time()
                q_lengths = [min(len(x), self.FLAGS.question_size) for x in q]
                p_lengths = [min(len(x), self.FLAGS.output_size) for x in p]
                q = [question[:self.FLAGS.question_size] + [PAD_ID] * (self.FLAGS.question_size - min(len(question), self.FLAGS.question_size)) for question in q]
                p = [paragraph[:self.FLAGS.output_size] + [PAD_ID] * (self.FLAGS.output_size - min(len(paragraph), self.FLAGS.output_size)) for paragraph in p]
                _, loss, grad_norm = self.optimize(session, q, p, a, q_lengths, p_lengths)
                num_processed += 1
                toc = time.time()
                if (num_processed % self.FLAGS.print_every == 0):
                    logging.info("Epoch = %d | Num batches processed = %d | Train epoch ETA = %f | Grad norm = %f | Training loss = %f" % (e, num_processed, (len(dataset) - num_processed) * (toc - tic), grad_norm, np.mean(loss)))
                
            # save the model
            num_processed = 0
            saver = tf.train.Saver()
            results_path = os.path.join(self.FLAGS.train_dir, "results/{:%Y%m%d_%H%M%S}/".format(datetime.datetime.now()))
            model_path = results_path + "model.weights/"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            save_path = saver.save(session, model_path)
            logging.info("Model saved in file: %s" % save_path)
            logging.info("Evaluating epoch %d", e)
            val_loss = self.validate(session, val_dataset, log=True)
            self.evaluate_answer(session, dataset, "Train", rev_vocab, log=True)
            self.evaluate_answer(session, val_dataset, "Validation", rev_vocab, log=True)
