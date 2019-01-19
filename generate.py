import os
import sys
import json

import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import hyparams as hp
import tacotron

from dataset import vctk, ljspeech, kss
from utils import script_process as sp
from utils import audio_process as ap
from utils import korean


def capture_alignments(alignments):
    dx, dy = 1, 1
    enc_steps, dec_steps = np.mgrid[
        slice(0, np.shape(alignments)[0] + dx, dx), slice(0, np.shape(alignments)[1] + dy, dy)
    ]
    fig = plt.figure()
    plt.pcolormesh(enc_steps, dec_steps, alignments)
    plt.xlabel("encoder steps")
    plt.ylabel("decoder steps")
    plt.colorbar()
    plt.tight_layout()
    fig.savefig(hp.logdir_root + "/%s-alignments-generated.png" % hp.dataset_name)
    plt.clf()
    plt.cla()
    plt.close()


def capture_spectrogram(spectrogram, name):
    fig = plt.figure()
    librosa.display.specshow(
        spectrogram,
        y_axis="linear",
        x_axis="time",
        sr=hp.sampling_rate,
        hop_length=hp.hop_length
    )
    plt.colorbar()
    plt.tight_layout()
    fig.savefig(hp.logdir_root + "/%s-%s-generated.png" % (hp.dataset_name, name))
    plt.clf()
    plt.cla()
    plt.close()


def main():
    os.environ["CUDA_DEVICE_ORDER"] = hp.cuda_device_order
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda_visible_devices

    batch_size = 1

    language = "kor" if hp.dataset_name == "kss" else "eng"

    with open("./dataset/%s_vocab.json" % hp.dataset_name, "r") as f:
        idx2char = json.load(f)
    char2idx = {char: idx for idx, char in enumerate(idx2char)}

    script_in = []
    print("type a script:")
    script_in.append(input())

    print(script_in)

    script_in = [sp.script_normalize(script, language=language) for script in script_in]
    print(script_in)
    script_in = [sp.encode_script(script, char2idx=char2idx) for script in script_in]

    print(script_in)

    scripts = tf.placeholder(shape=[batch_size, None], dtype=tf.int32)
    one_hot_scripts = sp.one_hot(scripts, len(idx2char))

    model = tacotron.Tacotron(
        batch_size=batch_size,
        multi_spk=hp.multi_speaker,
        char_emb_channels=hp.char_embedding_channels,
        enc_params=hp.enc_params,
        dec_params=hp.dec_params,
        post_cbhg_params=hp.post_cbhg_params,
        lin_out_channels=hp.linear_output_channels
    )
    mel_outs, lin_outs, dec_fin_states = model.create_model(
        inputs=one_hot_scripts, training=False)

    alignments = tf.transpose(dec_fin_states[0].alignment_history.stack(), [1, 2, 0])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("Restoring checkpoint to {} ...".format(hp.model_load_path), end="")
    saver = tf.train.Saver(max_to_keep=1000)
    saver.restore(sess, hp.model_load_path)
    print("Done...")

    print("Start generating...", end="")
    run_mel_outs, run_lin_outs, run_alignments = \
        sess.run([mel_outs, lin_outs, alignments], feed_dict={scripts: script_in})
    print("Done...")

    print(np.shape(run_lin_outs))

    run_mel_outs = run_mel_outs[0]
    run_lin_outs = run_lin_outs[0]
    run_alignments = run_alignments[0]

    audio_generated = ap.lin_to_audio(run_lin_outs)
    librosa.output.write_wav(
        hp.audio_save_path + "/audio_generated.wav", audio_generated, hp.sampling_rate)

    capture_spectrogram(spectrogram=run_lin_outs, name="lin")
    capture_spectrogram(spectrogram=run_mel_outs, name="mel")
    capture_alignments(alignments=run_alignments)


if __name__ == '__main__':
    main()
