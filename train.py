import os
import sys

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import hyparams as hp
import tacotron
from dataset import vctk, ljspeech, kss
from utils import script_process as sp


def learning_rate_decay(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme from tensor2tensor'''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


def capture_alignments(alignments, step):
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
    fig.savefig(hp.alignments_path + "/%s-alignments-%d.png" % (hp.dataset_name, step))
    plt.clf()
    plt.cla()
    plt.close()


def main():
    os.environ["CUDA_DEVICE_ORDER"] = hp.cuda_device_order
    os.environ["CUDA_VISIBLE_DEVICES"] = hp.cuda_visible_devices

    if hp.dataset_name == "vctk":
        loader = vctk.VCTKLoader(dataset_path=hp.dataset_path, batch_size=hp.batch_size)
    elif hp.dataset_name == "ljspeech":
        loader = ljspeech.LJSpeechLoader(dataset_path=hp.dataset_path, batch_size=hp.batch_size)
    elif hp.dataset_name == "kss":
        loader = kss.KSSLoader(dataset_path=hp.dataset_path, batch_size=hp.batch_size)
    print(loader.get_char_dict()[0])

    if hp.multi_speaker:
        mel_targets, lin_targets, spks, scripts = loader.create_dataset(use_tfrecord=True)
        one_hot_spks = sp.one_hot(spks, loader.get_num_spk(), squeeze=True)
    else:
        mel_targets, lin_targets, scripts = loader.create_dataset(use_tfrecord=True)
        one_hot_spks = None

    one_hot_scripts = sp.one_hot(scripts, loader.get_char_embedding_cardinality())

    model = tacotron.Tacotron(
        batch_size=hp.batch_size,
        multi_spk=hp.multi_speaker,
        char_emb_channels=hp.char_embedding_channels,
        enc_params=hp.enc_params,
        dec_params=hp.dec_params,
        post_cbhg_params=hp.post_cbhg_params,
        lin_out_channels=hp.linear_output_channels
    )
    mel_outs, lin_outs, dec_fin_states = model.create_model(
        inputs=one_hot_scripts, training=True, spks=one_hot_spks, dec_targets=mel_targets)

    alignments = tf.transpose(dec_fin_states[0].alignment_history.stack(), [1, 2, 0])

    mel_loss = tf.reduce_mean(tf.abs(mel_outs - mel_targets))
    lin_loss = tf.reduce_mean(tf.abs(lin_outs - lin_targets))

    loss = mel_loss + lin_loss

    global_step = tf.Variable(hp.start_step, name="global_step", trainable=False)
    learning_rate = learning_rate_decay(init_lr=hp.learning_rate, global_step=global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    gvs = optimizer.compute_gradients(loss)
    clipped = []
    for grad, var in gvs:
        grad = tf.clip_by_norm(grad, 5.0) if hp.grad_clipping else grad
        clipped.append((grad, var))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt = optimizer.apply_gradients(clipped, global_step=global_step)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    model_name = "%s-model.ckpt" % hp.dataset_name
    save_path = hp.model_save_path
    checkpoint_path = os.path.join(save_path, model_name)

    if hp.start_step == 0:
        print("Storing checkpoint to {} ...".format(save_path), end="")
        sys.stdout.flush()

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        saver = tf.train.Saver(max_to_keep=1000)
        saver.save(sess, checkpoint_path, global_step=0)
        print("Done.")
    else:
        print("Restoring checkpoint to {} ...".format(hp.model_load_path), end="")
        saver = tf.train.Saver(max_to_keep=1000)
        saver.restore(sess, hp.model_load_path)
        print("Done...")

    run_alignments = sess.run(alignments)
    capture_alignments(run_alignments[0], step=hp.start_step)

    for _ in range(hp.start_step, hp.final_training_step):
        step, run_loss, run_mel_loss, run_lin_loss, run_scripts, run_alignments, _ = \
            sess.run([global_step, loss, mel_loss, lin_loss, scripts, alignments, opt])
        print(
            "step: " + str(step) +
            " | loss = {0:.4f}".format(run_loss) +
            " | mel_loss = {0:.4f}".format(run_mel_loss) +
            " | lin_loss = {0:.4f}".format(run_lin_loss)
        )
        if step % hp.save_interval == 0:
            print("Storing checkpoint to {} ...".format(save_path), end="")
            sys.stdout.flush()

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            saver = tf.train.Saver(max_to_keep=1000)
            saver.save(sess, checkpoint_path, global_step=step)
            print("Done.")

            capture_alignments(run_alignments[0], step=step)


if __name__ == '__main__':
    main()
