# for a GPU setting
cuda_device_order = "PCI_BUS_ID"
cuda_visible_devices = "2"


# for general
dataset_name = "kss" # can be one of ["vctk", "ljspeech", "kss"]


# for audios
sampling_rate = 16000
frame_shift = 12.5e-3
preemphasis = 0.97
n_fft = 2048
n_mels = 80
n_freq = 1025
reduction_factor = 5
freq_threshold = 100
stft_window = "hann"
some_levels = {
    "vctk": (140, 20, 140, 20),
    "ljspeech": (120, 15, 140, 30),
    "kss": (140, 20, 140, 20)
}
mel_min_level, mel_ref_level, lin_min_level, lin_ref_level = some_levels[dataset_name]

hop_length = int(sampling_rate * frame_shift)
# shape of mel_spectrogram = [n_mels, mel_length]
# shape of stft = [n_freq, mel_length]


# for scripts
vocab = " abcdefghijklmnopqrstuvwxyz'.?"


# for training
logdir_root = "/hd2/hcnoh/multi-tacotron/log"
model_save_path = "/hd2/hcnoh/multi-tacotron/models"
alignments_path = "/hd2/hcnoh/multi-tacotron/alignments"

dataset_paths = {
    "vctk": "/hd2/hcnoh/dataset/VCTK/VCTK-Corpus",
    "ljspeech": "/hd2/hcnoh/dataset/LJSpeech-1.1",
    "kss": "/hd2/hcnoh/dataset/kss"
}
dataset_path = dataset_paths[dataset_name]

tfrecord_num = 256
tfrecord_paths = ["/hd2/hcnoh/dataset/tfrecord/%s(reduction_factor=%d).tfrecord-%d"
                  % (dataset_name, reduction_factor, i) for i in range(tfrecord_num)]

learning_rate = 0.001
start_step = 97000
final_training_step = 3000000
save_interval = 1000

grad_clipping = True

batch_size = 32


# for generating
model_load_path = "/hd2/hcnoh/multi-tacotron/models/%s-model.ckpt-382000" % dataset_name
audio_save_path = "/hd2/hcnoh/multi-tacotron/generated-audio"

max_decoder_length = 100
griffin_lim_iters = 60
power = 1.5


# Model parameters
multi_speaker = False

char_embedding_channels = 256

enc_cbhg_params = {
    "conv_bank_channels": 128,
    "conv_bank_K": 16,
    "conv_proj_channels": [128, 128],
    "conv_proj_kernel_sizes": [3, 3],
    "highway_units": 128,
    "gru_cells": 128 # output_channels == 256
}

enc_prenet_params = {"sizes": [256, 128], "dropout_rate": 0.5}

enc_params = {"prenet_params": enc_prenet_params, "cbhg_params": enc_cbhg_params}

dec_prenet_params = {"sizes": [256, 128], "dropout_rate": 0.5}

dec_params = {
    "attn_depth": 256,
    "attn_cells": 256,
    "prenet_params": dec_prenet_params,
    "res_cells": 256,
    "reduction_factor": reduction_factor,
    "output_channels": n_mels
}


# PostNet parameters
post_cbhg_params = {
    "conv_bank_channels": 128,
    "conv_bank_K": 8,
    "conv_proj_channels": [256, n_mels],
    "conv_proj_kernel_sizes": [3, 3],
    "highway_units": 128,
    "gru_cells": 128
}

linear_output_channels = n_freq
