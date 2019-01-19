import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl

from modules.networks import create_spk_emb, create_prenet


class ConcatWithAttentionWrapper(rnn_cell_impl.RNNCell):

    def __init__(self,
                 cell):
        super(ConcatWithAttentionWrapper).__init__()
        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size + self._cell.state_size.attention

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope=None):
        outputs, new_state = self._cell(inputs, state)
        attention = new_state.attention

        return tf.concat([outputs, attention], axis=-1), new_state


class PrenetWithReductionWrapper(rnn_cell_impl.RNNCell):

    def __init__(
        self,
        cell,
        training,
        reduction_channels,
        spks,
        prenet_name,
        prenet_params
    ):
        super(PrenetWithReductionWrapper).__init__()
        self._cell = cell
        self._training = training
        self._reduction_channels = reduction_channels
        self._spks = spks
        self._prenet_name = prenet_name
        self._prenet_params = prenet_params

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope=None):
        # shape of inputs = [batch_size, reduction_factor * reduction_channels]
        reduced_inputs = inputs[:, -self._reduction_channels:]
        if self._spks is not None:
            reduced_inputs = tf.concat([
                reduced_inputs,
                create_spk_emb(inputs=self._spks, units=self._reduction_channels)
            ], axis=-1)

        prenets = create_prenet(
            inputs=reduced_inputs,
            name=self._prenet_name,
            training=self._training,
            **self._prenet_params
        )
        outputs, new_state = self._cell(prenets, state)

        return (outputs, new_state)
