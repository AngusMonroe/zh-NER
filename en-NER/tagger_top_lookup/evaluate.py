#!/usr/bin/env python

import os
import numpy as np
import optparse
import itertools
from collections import OrderedDict
from evaluate_util import create_input
import loader

from evaluate_util import models_path, evaluate, eval_script, eval_temp
from loader import word_mapping, char_mapping, tag_mapping
from loader import update_tag_scheme, prepare_dataset
from loader import augment_with_pretrained
from model import Model

# Read parameters from command line
optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="dataset/train",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="dataset/dev",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="dataset/test",
    help="Test set location"
)
optparser.add_option(
    "-s", "--tag_scheme", default="iobes",
    help="Tagging scheme (IOB or IOBES)"
)
optparser.add_option(
    "-l", "--lower", default="0",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "-z", "--zeros", default="0",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "-c", "--char_dim", default="25",
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    "-C", "--char_lstm_dim", default="25",
    type='int', help="Char LSTM hidden layer size"
)
optparser.add_option(
    "-b", "--char_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for chars"
)
optparser.add_option(
    "-w", "--word_dim", default="100",
    type='int', help="Token embedding dimension"
)
optparser.add_option(
    "-W", "--word_lstm_dim", default="100",
    type='int', help="Token LSTM hidden layer size"
)
optparser.add_option(
    "-B", "--word_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)
optparser.add_option(
    "-p", "--pre_emb", default="",
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-A", "--all_emb", default="0",
    type='int', help="Load all embeddings"
)
optparser.add_option(
    "-a", "--cap_dim", default="0",
    type='int', help="Capitalization feature dimension (0 to disable)"
)
optparser.add_option(
    "-f", "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
optparser.add_option(
    "-L", "--lr_method", default="sgd-lr_.005",
    help="Learning method (SGD, Adadelta, Adam..)"
)
optparser.add_option(
    "-r", "--reload", default="1",
    type='int', help="Reload the last saved model"
)
opts = optparser.parse_args()[0]

# Parse parameters
parameters = OrderedDict()
parameters['tag_scheme'] = opts.tag_scheme
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['char_dim'] = opts.char_dim
parameters['char_lstm_dim'] = opts.char_lstm_dim
parameters['char_bidirect'] = opts.char_bidirect == 1
parameters['word_dim'] = opts.word_dim
parameters['word_lstm_dim'] = opts.word_lstm_dim
parameters['word_bidirect'] = opts.word_bidirect == 1
parameters['pre_emb'] = opts.pre_emb
parameters['all_emb'] = opts.all_emb == 1
parameters['cap_dim'] = opts.cap_dim
parameters['crf'] = opts.crf == 1
parameters['dropout'] = opts.dropout
parameters['lr_method'] = opts.lr_method

# Check parameters validity
print(opts.train)
assert os.path.isfile(opts.train)
assert os.path.isfile(opts.dev)
assert os.path.isfile(opts.test)
assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['tag_scheme'] in ['iob', 'iobes']
assert not parameters['all_emb'] or parameters['pre_emb']
assert not parameters['pre_emb'] or parameters['word_dim'] > 0
assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

# Check evaluation script / folders
if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)

# Initialize model
model = Model(parameters=parameters, models_path=models_path)
print "Model location: %s" % model.model_path

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']
# Load sentences
train_sentences = loader.load_sentences(opts.train, lower, zeros)
# Use selected tagging scheme (IOB / IOBES)
update_tag_scheme(train_sentences, tag_scheme)
# Create a dictionary / mapping of words
# If we use pretrained embeddings, we add them to the dictionary.
dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
dico_words_train = dico_words
# Create a dictionary and a mapping for words / POS tags / tags
dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)


# Save the mappings to disk
print 'Saving the mappings to disk...'
# how this work, should have a mapping in disk and load every time?
model.save_mappings(id_to_word, id_to_char, id_to_tag)

# Build the model
f_train, f_eval = model.build(**parameters)

# Reload previous model values
if opts.reload:
    print 'Reloading previous model...'
    model.reload()

def get_ne(sentence):
    test_sentences = loader.load_test_sentence(sentence, lower, zeros)
    test_data = prepare_dataset(
        test_sentences, word_to_id, char_to_id, lower
    )
    return evaluate(parameters, f_eval, test_sentences,
            test_data, id_to_tag, dico_tags)

if __name__ == '__main__':
    while(True):
        sentence = raw_input("input >>> ")
        test_sentences = loader.load_test_sentence(sentence, lower, zeros)
        test_data = prepare_dataset(
            test_sentences, word_to_id, char_to_id, lower
        )
        print evaluate(parameters, f_eval, test_sentences,
                test_data, id_to_tag, dico_tags)

