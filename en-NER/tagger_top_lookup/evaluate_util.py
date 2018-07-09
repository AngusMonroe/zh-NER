import os
import re
import codecs
import numpy as np
import theano
from lookup_util import look_up


models_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")


def get_name(parameters):
    """
    Generate a model name from its parameters.
    """
    l = []
    for k, v in parameters.items():
        if type(v) is str and "/" in v:
            l.append((k, v[::-1][:v[::-1].index('/')][::-1]))
        else:
            l.append((k, v))
    name = ",".join(["%s=%s" % (k, str(v).replace(',', '')) for k, v in l])
    return "".join(i for i in name if i not in "\/:*?<>|")


def set_values(name, param, pretrained):
    """
    Initialize a network parameter with pretrained values.
    We check that sizes are compatible.
    """
    param_value = param.get_value()
    if pretrained.size != param_value.size:
        raise Exception(
            "Size mismatch for parameter %s. Expected %i, found %i."
            % (name, param_value.size, pretrained.size)
        )
    param.set_value(np.reshape(
        pretrained, param_value.shape
    ).astype(np.float32))


def shared(shape, name):
    """
    Create a shared object of a numpy array.
    """
    if len(shape) == 1:
        value = np.zeros(shape)  # bias are initialized with zeros
    else:
        drange = np.sqrt(6. / (np.sum(shape)))
        value = drange * np.random.uniform(low=-1.0, high=1.0, size=shape)
    return theano.shared(value=value.astype(theano.config.floatX), name=name)


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


def pad_word_chars(words):
    """
    Pad the characters of the words in a sentence.
    Input:
        - list of lists of ints (list of words, a word being a list of char indexes)
    Output:
        - padded list of lists of ints
        - padded list of lists of ints (where chars are reversed)
        - list of ints corresponding to the index of the last character of each word
    """
    max_length = max([len(word) for word in words])
    char_for = []
    char_rev = []
    char_pos = []
    for word in words:
        padding = [0] * (max_length - len(word))
        char_for.append(word + padding)
        char_rev.append(word[::-1] + padding)
        char_pos.append(len(word) - 1)
    return char_for, char_rev, char_pos


def create_input(data, parameters, add_label, singletons=None):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    words = data['words']
    chars = data['chars']
    if singletons is not None:
        words = insert_singletons(words, singletons)
    if parameters['cap_dim']:
        caps = data['caps']
    char_for, char_rev, char_pos = pad_word_chars(chars)
    input = []
    if parameters['word_dim']:
        input.append(words)
    if parameters['char_dim']:
        input.append(char_for)
        if parameters['char_bidirect']:
            input.append(char_rev)
        input.append(char_pos)
    if parameters['cap_dim']:
        input.append(caps)
    if add_label:
        input.append(data['tags'])
    return input

def parse(tags):
    filter_ = []
    current_str = ''
    last_tag = ''
    for raw, tag in tags:
        if tag == 'O':
            if len(current_str) > 0:
                filter_.append((current_str, last_tag))
                current_str = ''
                last_tag = ''
            continue
        elif tag[2:] == last_tag:
            current_str += ' ' + raw
        else:
            if len(current_str) > 0:
                filter_.append((current_str, last_tag))
            last_tag = tag[2:]
            current_str = raw
    if len(current_str) > 0:
        filter_.append((current_str, last_tag))
    return filter_

def norm(sentences, score_k):
    def correct(sen):
        pre = ''
        new = []
        for raw, tag in sen:
            if tag.startswith('I'):
# O, I-org or B-org, I-key
                if tag[2:] != pre:
                    tag = 'B-' + tag[2:] 
            pre = tag[2:]
            new.append((raw, tag))
        return new 

    def hash_(sen):
        return ''.join(zip(*sen)[1])

    sen_set = set()
    final = []
    hash_final = []
    scores = {}
    for i, sen in enumerate(sentences):
        sen = correct(sen)
        sentences[i] = sen
        if hash_(sen) in scores:
            scores[hash_(sen)] += score_k[i]
        else:
            scores[hash_(sen)] = score_k[i]
        if hash_(sen) not in sen_set:
            sen_set.add(hash_(sen))
            final.append(sen)
    num_scores = []
    for f in final:
        num_scores.append(scores[hash_(f)])
    num_scores = num_scores / sum(num_scores)
    assert len(final) == len(num_scores)
    tup = zip(final, num_scores)
    tup = sorted(tup, key = lambda x: x[1], reverse = True)
    return tup 

def evaluate(parameters, f_eval, raw_sentences, parsed_sentences,
             id_to_tag, dictionary_tags):
    """
    Evaluate current model using CoNLL script.
    """
    n_tags = len(id_to_tag)
    predictions = []
    sentences = []
    count = np.zeros((n_tags, n_tags), dtype=np.int32)

    for raw_sentence, data in zip(raw_sentences, parsed_sentences):
        input = create_input(data, parameters, False)
        if parameters['crf']:
            res = f_eval(*input)
            y_preds_top_k = res[:-1]
            score_top_k = res[-1]
        else:
            y_preds = f_eval(*input).argmax(axis=1)

        for y_preds in y_preds_top_k:
            # y_preds = np.array(y_preds)[1:-1]
            y_preds = np.array(y_preds)
            y_preds = y_preds[1:-1]
# hard code
            p_tags = [id_to_tag[y_pred % 20] for y_pred in y_preds]
            if parameters['tag_scheme'] == 'iobes':
               p_tags = iobes_iob(p_tags)
            lines = []
            sentence = []
            for i, y_pred in enumerate(y_preds):
                new_line = " ".join([raw_sentence[i], p_tags[i]])
                lines.append(new_line)
                sentence.append((raw_sentence[i], p_tags[i])) 
            sentences.append(sentence)
            predictions.append(' '.join(lines))
        sentences = norm(sentences, score_top_k)
        results = []
        oriresults = []
        for sen, conf in sentences:
            result = '{}, probability: {}'.format(parse(sen), conf)
            results.append(result)
            oriresults.append((parse(sen), conf))

    #    print (np.array(score_top_k))
    uniq_dic, dic = look_up(oriresults)
    return uniq_dic, dic, results 
