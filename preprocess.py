import hyparams as hp
from dataset import vctk, ljspeech, kss
from utils import script_process as sp


def main():
    if hp.dataset_name == "vctk":
        loader = vctk.VCTKLoader(dataset_path=hp.dataset_path, batch_size=hp.batch_size)
    elif hp.dataset_name == "ljspeech":
        loader = ljspeech.LJSpeechLoader(dataset_path=hp.dataset_path, batch_size=hp.batch_size)
    elif hp.dataset_name == "kss":
        loader = kss.KSSLoader(dataset_path=hp.dataset_path, batch_size=hp.batch_size)
    print(loader.get_char_dict()[0])
    dataset_list = loader.get_dataset_list()

    def _get_period(length, num):
        period = int(length / (num + 1)) + 1
        remains = length - num * period
        while remains > 0:
            if length == num * period + remains:
                return period, remains
            else:
                period += 1
                remains = length - num * period
        raise ValueError()

    period = int(len(dataset_list) / (hp.tfrecord_num - 1))
    remains = len(dataset_list) % (hp.tfrecord_num - 1)
    chunked_dataset_list = [dataset_list[i * period: i * period + period]
                            for i in range(hp.tfrecord_num - 1)]
    chunked_dataset_list.append(dataset_list[-remains:])

    for dataset_list, tfrecord_path in zip(chunked_dataset_list, hp.tfrecord_paths):
        loader.create_tfrecord(dataset_list, tfrecord_path)

    mel_targets, lin_targets, spks, scripts = loader.create_dataset(use_tfrecord=True)

    import tensorflow as tf
    import numpy as np

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    mel_targets_, lin_targets_, scripts_ = sess.run([
        mel_targets, lin_targets, scripts
    ])

    print(np.shape(mel_targets_))
    print(np.shape(lin_targets_))
    #print(np.shape(spks_))
    print(np.shape(scripts_))
    #print(spks_)
    print(scripts_)


if __name__ == '__main__':
    main()
