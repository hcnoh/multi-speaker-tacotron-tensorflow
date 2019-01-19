import os
import sys
import json
import random

import numpy as np
import tensorflow as tf

import hyparams as hp
from utils import script_process as sp
from utils import audio_process as ap


def get_dataset_list(dataset_path):
    audio_file_names = []
    audio_full_paths = {}
    script_file_names = []
    scripts = {}
    dataset_list = []
    for [path_, dir_, files_] in os.walk(dataset_path):
        for file_name in files_:
            ext = os.path.splitext(file_name)[-1]
            if ext == ".wav":
                full_file_name = path_ + "/" + file_name
                full_file_name_ = full_file_name.split("/")
                file_name_ = full_file_name_[-1].replace(".wav", "")
                audio_full_paths[file_name_] = full_file_name
                audio_file_names.append(file_name_)
            elif ext == ".txt":
                full_file_name = path_ + "/" + file_name
                full_file_name_ = full_file_name.split("/")
                file_name_ = full_file_name_[-1].replace(".txt", "")
                scripts[file_name_] = get_script(full_file_name)
                script_file_names.append(file_name_)

    audio_file_names = set(audio_file_names)
    script_file_names = set(script_file_names)
    file_names = list(audio_file_names & script_file_names)
    file_names.sort()
    for file_name in file_names:
        dataset_list.append({
            "audio_file_path": audio_full_paths[file_name],
            "spk": file_name[:4],
            "script": scripts[file_name]
        })

    return dataset_list


def get_script(script_file_path):
    script = []
    with open(file=script_file_path, mode="rt", encoding="utf-8") as f:
        for line in f:
            script.append(sp.script_normalize(line.lower()))
    script = list("".join(script))

    return script


def get_idx2char(dataset_list):
    idx2char = set()
    scripts = [dataset["script"] for dataset in dataset_list]
    for script in scripts:
        idx2char.update(set(script))
    idx2char = list(idx2char)
    idx2char.append("P") # P: Padding
    idx2char.sort()
    char2idx = {char: idx for idx, char in enumerate(idx2char)}

    with open("./dataset/vctk_vocab.json", "w") as f:
        json.dump(idx2char, f, indent=4)

    return idx2char, char2idx


def get_idx2spk(dataset_list):
    idx2spk = set()
    spks = [dataset["spk"] for dataset in dataset_list]
    for spk in spks:
        idx2spk.update({spk})
    idx2spk = list(idx2spk)
    idx2spk.sort()
    spk2idx = {spk: idx for idx, spk in enumerate(idx2spk)}

    return idx2spk, spk2idx


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def from_tfrecord(serialized):
    features = tf.parse_single_example(
        serialized=serialized,
        features={
            "mel": tf.FixedLenFeature([], tf.string),
            "lin": tf.FixedLenFeature([], tf.string),
            "spk": tf.FixedLenFeature([], tf.string),
            "script": tf.FixedLenFeature([], tf.string)
        }
    )
    mel = tf.reshape(tf.decode_raw(features["mel"], tf.float32), [hp.n_mels, -1])
    lin = tf.reshape(tf.decode_raw(features["lin"], tf.float32), [hp.n_freq, -1])
    spk = tf.decode_raw(features["spk"], tf.int32)
    script = tf.decode_raw(features["script"], tf.int32)

    return mel, lin, spk, script


class VCTKLoader(object):

    def __init__(self, dataset_path, batch_size):
        self._dataset_path = dataset_path
        self._dataset_list = get_dataset_list(self._dataset_path)
        self._batch_size = batch_size
        self._num_data = len(self._dataset_list)
        self._idx2char, self._char2idx = get_idx2char(self._dataset_list)
        self._idx2spk, self._spk2idx = get_idx2spk(self._dataset_list)
        self._char_embedding_cardinality = len(self._idx2char)
        self._num_spk = len(self._idx2spk)

    def get_char_dict(self):
        return self._idx2char, self._char2idx

    def get_spk_dict(self):
        return self._idx2spk, self._spk2idx

    def get_char_embedding_cardinality(self):
        return self._char_embedding_cardinality

    def get_num_spk(self):
        return self._num_spk

    def create_tfrecord(self, dataset_list, tfrecord_path):
        print("Start converting...")
        options = tf.python_io.\
            TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)
        writer = tf.python_io.TFRecordWriter(path=tfrecord_path, options=options)

        def _round_up(length, reduction_factor):
            remain = length % reduction_factor
            return reduction_factor - remain

        for dataset in dataset_list:
            audio_file_path = dataset["audio_file_path"]

            _, mel, lin = ap.get_features(audio_file_path)
            padding_length = _round_up(np.shape(mel)[-1], hp.reduction_factor)
            mel = np.pad(mel, ((0, 0), (0, padding_length)), "constant", constant_values=0.)
            lin = np.pad(lin, ((0, 0), (0, padding_length)), "constant", constant_values=0.)

            spk = np.asarray(self._spk2idx[dataset["spk"]], dtype=np.int32)

            script = dataset["script"]
            script = sp.encode_script(script, char2idx=self._char2idx)
            script = np.asarray(script, dtype=np.int32)

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "mel": _bytes_feature(mel.tostring()),
                        "lin": _bytes_feature(lin.tostring()),
                        "spk": _bytes_feature(spk.tostring()),
                        "script": _bytes_feature(script.tostring())
                    }
                )
            )
            writer.write(example.SerializeToString())

        writer.close()
        print("Done...")

    def _generate_batch(self):
        def round_up(length, reduction_factor):
            remain = length % reduction_factor
            return reduction_factor - remain

        while True:
            random_dataset_list = self._dataset_list[:]
            random.shuffle(random_dataset_list)
            for dataset in random_dataset_list:
                audio_file_path = dataset["audio_file_path"]

                _, mel, lin = ap.get_features(audio_file_path)
                padding_length = round_up(np.shape(mel)[-1], hp.reduction_factor)
                mel = np.pad(mel, ((0, 0), (0, padding_length)), "constant", constant_values=0.)
                lin = np.pad(lin, ((0, 0), (0, padding_length)), "constant", constant_values=0.)

                spk = np.asarray(self._spk2idx[dataset["spk"]], dtype=np.int32)

                script = dataset["script"]
                script = sp.encode_script(script, char2idx=self._char2idx)
                script = np.asarray(script, dtype=np.int32)

                yield mel, lin, spk, script

    def create_dataset(self, use_tfrecord=True):
        if use_tfrecord:
            self._dataset = tf.data.TFRecordDataset(
                filenames=hp.tfrecord_paths, compression_type="GZIP")

        else:
            self._dataset = tf.data.Dataset.from_generator(
                generator=self._generate_batch,
                output_types=(tf.float32, tf.float32, tf.int32, tf.int32),
                output_shapes=(
                    tf.TensorShape([hp.n_mels, None]),
                    tf.TensorShape([hp.n_freq, None]),
                    tf.TensorShape([]),
                    tf.TensorShape([None])
                )
            )

        return self._dataset.\
            map(from_tfrecord, num_parallel_calls=4).\
            apply(tf.contrib.data.shuffle_and_repeat(
                self._batch_size * 100, hp.final_training_step)).\
            prefetch(self._batch_size).\
            padded_batch(
                batch_size=self._batch_size,
                padded_shapes=(
                    tf.TensorShape([hp.n_mels, None]),
                    tf.TensorShape([hp.n_freq, None]),
                    tf.TensorShape([1]),
                    tf.TensorShape([None])
                ),
                padding_values=(0., 0., 0, self._char2idx["P"]),
                drop_remainder=True).\
            make_one_shot_iterator().get_next()

    def get_dataset_list(self):
        return self._dataset_list
