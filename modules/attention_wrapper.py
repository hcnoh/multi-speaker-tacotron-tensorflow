import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import AttentionWrapper
from tensorflow.python.framework import dtypes, ops
from tensorflow.python.ops import array_ops, math_ops, rnn_cell_impl, nn_ops, variable_scope
from tensorflow.python.layers import core

from modules.networks import create_spk_emb


def _get_score(processed_query, keys, condition):
    dtype = processed_query.dtype
    num_units = keys.shape[2].value or array_ops.shape(keys)[2]
    processed_query = array_ops.expand_dims(processed_query, 1)
    v = variable_scope.get_variable(
        "attention_v", [num_units], dtype=dtype)

    b = tf.expand_dims(create_spk_emb(inputs=condition, units=num_units), axis=1)\
    if condition is not None else 0

    return math_ops.reduce_sum(v * math_ops.tanh(keys + processed_query + b), [2])


_zero_state_tensors = rnn_cell_impl._zero_state_tensors


class ConditionalBahdanauAttention(BahdanauAttention):

    def __init__(
        self,
        num_units,
        condition,
        memory,
        memory_sequence_length=None,
        probability_fn=None,
        score_mask_value=None,
        dtype=None,
        name="ConditionalBahdanauAttention"
    ):
        super(ConditionalBahdanauAttention, self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            normalize=False,
            probability_fn=probability_fn,
            score_mask_value=score_mask_value,
            dtype=dtype,
            name=name
        )
        self._condition = condition

    def __call__(self, query, state):
        if self._condition is not None:
            with variable_scope.variable_scope(None, "conditional_bahdanau_attention", [query]):
                processed_query = self.query_layer(query) if self.query_layer else query
                score = _get_score(processed_query, self._keys, self._condition)
            alignments = self._probability_fn(score, state)
            next_state = alignments

            return alignments, next_state
        else:
            return super(ConditionalBahdanauAttention, self).__call__(query, state)


class ConditionalAttentionWrapper(AttentionWrapper):

    def __init__(
        self,
        condition,
        cell,
        attention_mechanism,
        attention_layer_size=None,
        alignment_history=False,
        cell_input_fn=None,
        output_attention=True,
        initial_cell_state=None,
        name=None,
        attention_layer=None
    ):
        super(ConditionalAttentionWrapper, self).__init__(
            cell=cell,
            attention_mechanism=attention_mechanism,
            attention_layer_size=attention_layer_size,
            alignment_history=alignment_history,
            cell_input_fn=cell_input_fn,
            output_attention=output_attention,
            initial_cell_state=initial_cell_state,
            name=name,
            attention_layer=attention_layer
        )
        self._condition = condition

    def zero_state(self, batch_size, dtype):
        if self._condition is not None:
            zero_state = super(ConditionalAttentionWrapper, self).zero_state(
                batch_size=batch_size, dtype=dtype)
            zero_state = zero_state._replace(
                attention=create_spk_emb(inputs=self._condition, units=self.state_size.attention))

            return zero_state
        else:
            return super(ConditionalAttentionWrapper, self).zero_state(
                batch_size=batch_size, dtype=dtype)
