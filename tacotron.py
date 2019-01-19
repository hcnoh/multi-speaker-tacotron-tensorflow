import tensorflow as tf

import hyparams as hp
from modules.networks import create_char_emb, create_prenet, create_cbhg_layer
from modules import attention_wrappers, rnn_wrappers, helpers


def create_encoder(inputs, spks, training, prenet_params, cbhg_params):
    with tf.variable_scope("encoder"):
        # shape of inputs = [batch_size, sequence_length, char_embedding_cardinality]
        # for pre-net
        prenets = create_prenet(
            inputs=inputs, name="prenet", training=training, **prenet_params)

        # for CBHG
        outputs = create_cbhg_layer(
            inputs=prenets, spks=spks, name="cbhg", training=training, **cbhg_params)

    return outputs


def create_decoder(
    batch_size,
    training,
    targets,
    spks,
    attn_memory,
    attn_depth,
    attn_cells,
    prenet_params,
    res_cells,
    reduction_factor,
    output_channels
):
    with tf.variable_scope("decoder"):
        attn_cell = attention_wrappers.ConditionalAttentionWrapper(
            condition=spks,
            cell=tf.nn.rnn_cell.GRUCell(num_units=attn_cells, name="attn_cell"),
            attention_mechanism=attention_wrappers.ConditionalBahdanauAttention(
                num_units=attn_depth, condition=spks, memory=attn_memory),
            alignment_history=True,
            output_attention=False
        )

        attn_prenet_cell = rnn_wrappers.PrenetWithReductionWrapper(
            training=training,
            cell=attn_cell,
            reduction_channels=output_channels,
            spks=spks,
            prenet_name="attn_prenet",
            prenet_params=prenet_params
        )

        attn_concat_cell = rnn_wrappers.ConcatWithAttentionWrapper(cell=attn_prenet_cell)

        attn_proj_cell = tf.contrib.rnn.OutputProjectionWrapper(
            cell=attn_concat_cell, output_size=res_cells)

        dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            cells=[attn_proj_cell,
                   tf.nn.rnn_cell.GRUCell(num_units=res_cells, name="res_cell0"),
                   tf.nn.rnn_cell.GRUCell(num_units=res_cells, name="res_cell1")],
            state_is_tuple=True
        )

        dec_proj_cell = tf.contrib.rnn.OutputProjectionWrapper(
            cell=dec_cell, output_size=reduction_factor * output_channels)

        initial_states = dec_proj_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

        go_frame = tf.constant(
            value=0, dtype=tf.float32, shape=[reduction_factor * output_channels])

        helper = None
        max_decoder_length = None

        if training:
            chunked_targets = tf.reshape(
                targets, [batch_size, -1, reduction_factor * output_channels])

            helper = helpers.TrainingHelper(
                batch_size=batch_size,
                reduction_channels=output_channels,
                targets=chunked_targets,
                go_frame=go_frame
            )

        else:
            helper = helpers.GeneratingHelper(batch_size=batch_size, go_frame=go_frame)
            max_decoder_length = hp.max_decoder_length

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=dec_proj_cell, helper=helper, initial_state=initial_states)

        (dec_outs, _), dec_fin_states, dec_fin_seq_lengths = \
            tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, maximum_iterations=max_decoder_length)

        outputs = tf.reshape(dec_outs, [batch_size, -1, output_channels])

        return outputs, dec_fin_states


class Tacotron(object):

    def __init__(
        self,
        batch_size,
        multi_spk,
        char_emb_channels,
        enc_params,
        dec_params,
        post_cbhg_params,
        lin_out_channels
    ):
        self._multi_spk = multi_spk
        self._batch_size = batch_size
        self._char_emb_channels = char_emb_channels
        self._enc_params = enc_params
        self._dec_params = dec_params
        self._post_cbhg_params = post_cbhg_params
        self._lin_out_channels = lin_out_channels

    def create_model(self, inputs, training, spks=None, dec_targets=None):
        if self._multi_spk:
            assert spks is not None
        char_embs = create_char_emb(inputs=inputs, units=self._char_emb_channels)

        trans_dec_targets = None
        if training:
            assert dec_targets is not None
            trans_dec_targets = tf.transpose(dec_targets, perm=[0, 2, 1])

        enc_outs = create_encoder(
            inputs=char_embs,
            spks=spks,
            training=training,
            **self._enc_params
        )

        dec_outs, dec_fin_states = create_decoder(
            batch_size=self._batch_size,
            training=training,
            targets=trans_dec_targets,
            spks=spks,
            attn_memory=enc_outs,
            **self._dec_params
        )

        postnet_outs = create_cbhg_layer(
            inputs=dec_outs,
            spks=spks,
            name="post_cbhg",
            training=training,
            **self._post_cbhg_params
        )

        dense_outs = tf.layers.dense(
            inputs=postnet_outs, units=self._lin_out_channels, activation=None, name="post_dense")

        mel_outs = tf.transpose(dec_outs, [0, 2, 1])
        lin_outs = tf.transpose(dense_outs, [0, 2, 1])

        return mel_outs, lin_outs, dec_fin_states
