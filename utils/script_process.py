import unicodedata
import re

import numpy as np
import tensorflow as tf

import hyparams as hp

from utils import korean


def encode_script(script, char2idx):
    encoded = [char2idx[char] for char in script]

    return encoded


def decode_script(encoded, idx2char):
    decoded = "".join([idx2char[idx] for idx in encoded])

    return decoded


def one_hot(scripts, cardinality, squeeze=False):
    # shape of scripts = [batch_size, length]
    # Note that the scripts should be indexes, not characters
    one_hot = tf.one_hot(tf.cast(scripts, tf.int32), cardinality, axis=-1)

    return tf.squeeze(one_hot) if squeeze else one_hot


def inv_one_hot(one_hots):
    # shape of one_hots = [batch_size, length, depth]
    inv_one_hots = tf.argmax(one_hots, axis=-1)

    return inv_one_hots


def script_normalize(script, language="eng"):
    if language == "eng":
        '''
        from https://github.com/Kyubyong/tacotron/blob/master/data_load.py
        '''
        normalized_script = "".join(char for char in unicodedata.normalize("NFD", script)
                                    if unicodedata.category(char) != "Mn")
        normalized_script = normalized_script.lower()
        normalized_script = re.sub("[^{}]".format(hp.vocab), " ", normalized_script)
        normalized_script = re.sub("[ ]+", " ", normalized_script) + "E" # E: End of sentence

        return normalized_script

    elif language == "kor":
        normalized_script = korean.parse(script)
        normalized_script = "".join([korean.ord2sym[ord(char)] for char in normalized_script])

        print(script)
        print(normalized_script)

        return normalized_script
