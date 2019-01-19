import os
import sys
import random

import numpy as np
import tensorflow as tf
import librosa
import librosa.display

import matplotlib
matplotlib.use('agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d, Axes3D

import tacotron
import hyparams as hp
from dataset import kss
from utils import audio_process as ap
from utils import script_process as sp


def round_up(length, reduction_factor):
    remain = length % reduction_factor
    return reduction_factor - remain


def main():
    dataset_list = kss.get_dataset_list(hp.dataset_path)
    idx2char, char2idx, idx2ord = kss.get_idx2char(dataset_list)

    print(len(dataset_list))
    print(len(idx2char))
    print(idx2char)
    #print([(char.encode("utf-8"), char) for char in idx2char])
    #print([ord(char) for char in idx2char])
    #print([elem for elem in zip(idx2char, idx2ord)])

    random_dataset_list = dataset_list[:]
    random.shuffle(random_dataset_list)
    print([dataset["script"] for dataset in random_dataset_list[:5]])
    print([dataset["normalized_script"] for dataset in random_dataset_list[:5]])

    mels = []
    lins = []

    audio, mel, lin = ap.get_features(dataset_list[0]["audio_file_path"])
    audio_inverse = ap.lin_to_audio(lin)

    padding_length = round_up(np.shape(mel)[-1], hp.reduction_factor)
    mel = np.pad(mel, ((0, 0), (0, padding_length)), "constant", constant_values=0.)
    lin = np.pad(lin, ((0, 0), (0, padding_length)), "constant", constant_values=0.)

    print(dataset_list[0]["audio_file_path"])

    librosa.output.write_wav(
        "%s/audio_sample.wav" % hp.audio_save_path, audio, hp.sampling_rate
    )
    librosa.output.write_wav(
        "%s/audio_inverse_sample.wav" % hp.audio_save_path, audio_inverse, hp.sampling_rate
    )

    fig = plt.figure()
    librosa.display.specshow(lin,
                             y_axis="linear",
                             x_axis="time",
                             sr=hp.sampling_rate,
                             hop_length=hp.hop_length)
    plt.colorbar()
    plt.tight_layout()
    fig.savefig("%s/lin_sample.png" % hp.logdir_root)
    plt.clf()
    plt.cla()
    plt.close()

    fig = plt.figure()
    librosa.display.specshow(mel,
                             y_axis="linear",
                             x_axis="time",
                             sr=hp.sampling_rate,
                             hop_length=hp.hop_length)
    plt.colorbar()
    plt.tight_layout()
    fig.savefig("%s/mel_sample.png" % hp.logdir_root)
    plt.clf()
    plt.cla()
    plt.close()

    random.shuffle(dataset_list)

    for dataset in dataset_list[:10]:
        _, mel, lin = ap.get_features(dataset["audio_file_path"])
        padding_length = round_up(np.shape(mel)[-1], hp.reduction_factor)
        #mel = np.pad(mel, ((0, 0), (0, padding_length)), "constant", constant_values=0.)
        #lin = np.pad(lin, ((0, 0), (0, padding_length)), "constant", constant_values=0.)
        mel = np.reshape(mel, [-1]).tolist()
        lin = np.reshape(lin, [-1]).tolist()
        mels = mels + mel
        lins = lins + lin

    fig = plt.figure()
    n , bins, patches = plt.hist(mels, 50, density=1, facecolor="blue", alpha=0.75)
    fig.savefig("%s/hist_mel.png" % hp.logdir_root)

    fig = plt.figure()
    n , bins, patches = plt.hist(lins, 50, density=1, facecolor="blue", alpha=0.75)
    fig.savefig("%s/hist_lin.png" % hp.logdir_root)


if __name__ == '__main__':
    main()
