import tensorflow as tf


def create_char_emb(inputs, units):
    return tf.layers.dense(
        inputs=inputs,
        units=units,
        use_bias=False,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.5),
        name="char_emb"
    )


def create_spk_emb(inputs, units):
    return tf.layers.dense(
        inputs=inputs,
        units=units,
        activation=tf.nn.softsign,
        use_bias=False,
        kernel_initializer=tf.initializers.random_uniform(minval=-0.1, maxval=0.1),
        name="spk_emb"
    )


def create_prenet(inputs, name, training, sizes, dropout_rate):
    with tf.variable_scope(name):
        prenets = inputs
        for i, size in enumerate(sizes):
            prenets = tf.layers.dense(
                inputs=prenets, units=size, activation=tf.nn.relu, name="dense%d" % i)
            prenets = tf.layers.dropout(
                inputs=prenets, rate=dropout_rate, training=training, name="dropout%d" % i)

    return prenets


def create_cbhg_layer(
    inputs,
    spks,
    name,
    training,
    conv_bank_channels,
    conv_bank_K,
    conv_proj_channels,
    conv_proj_kernel_sizes,
    highway_units, gru_cells
):
    with tf.variable_scope(name):
        # shape of inputs: [batch_size, sequence_length, char_embedding_channels]

        with tf.variable_scope("conv_bank"):
            convs = [tf.layers.batch_normalization(
                inputs=tf.layers.conv1d(
                    inputs=inputs,
                    filters=conv_bank_channels,
                    kernel_size=i,
                    strides=1, padding="same",
                    activation=tf.nn.relu),
                training=training)
                for i in range(1, conv_bank_K + 1)
            ]

            conv_banks = tf.concat(convs, axis=-1)

            max_pools = tf.layers.max_pooling1d(
                inputs=conv_banks, pool_size=2, strides=1, padding="same")

        with tf.variable_scope("conv_proj"):
            conv_projs1 = tf.layers.batch_normalization(
                inputs=tf.layers.conv1d(
                    inputs=max_pools, filters=conv_proj_kernel_sizes[0],
                    kernel_size=conv_proj_kernel_sizes[0],
                    strides=1,
                    padding="same",
                    activation=tf.nn.relu),
                training=training
            )
            conv_projs2 = tf.layers.batch_normalization(
                inputs=tf.layers.conv1d(
                    inputs=conv_projs1,
                    filters=conv_proj_channels[1],
                    kernel_size=conv_proj_kernel_sizes[1],
                    strides=1,
                    padding="same",
                    activation=None),
                training=training
            )

        # for residual
        conv_projs2 = conv_projs2 + inputs
        # shape of residual_connections = [batch_size, sequence_length, conv_proj_channels[-1]]

        # for highway
        with tf.variable_scope("highway"):
            # should make the channels of residual_connections and highway_units same
            highways = conv_projs2
            if highway_units != conv_proj_channels[1]:
                highways = tf.layers.dense(
                    inputs=highways,
                    units=highway_units,
                    activation=None,
                    name="dense_for_highways"
                )

            if spks is not None:
                spk_embs = create_spk_emb(inputs=spks, units=highway_units)

            for i in range(4):
                if spks is not None:
                    highways = highways + tf.expand_dims(spk_embs, axis=1) # don't need to tile

                new_highways = tf.layers.dense(
                    inputs=highways,
                    units=highway_units,
                    activation=tf.nn.relu,
                    name="new_highways_%d" % i
                )
                transform_gate = tf.layers.dense(
                    inputs=highways,
                    units=highway_units,
                    activation=tf.nn.sigmoid,
                    bias_initializer=tf.constant_initializer(-1.),
                    name="transform_gate_%d" % i
                )
                highways = transform_gate * new_highways + (1. - transform_gate) * highways
        # shape of highways = [batch_size, sequence_length, highway_units]

        # for bidirectional-GRU
        with tf.variable_scope("bi_gru"):
            spk_embs = create_spk_emb(inputs=spks, units=gru_cells) if spks is not None else None

            gru_outs, gru_fin_states = \
                tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=tf.nn.rnn_cell.GRUCell(num_units=gru_cells, name="cell_fw"),
                    cell_bw=tf.nn.rnn_cell.GRUCell(num_units=gru_cells, name="cell_bw"),
                    inputs=highways,
                    initial_state_fw=spk_embs,
                    initial_state_bw=spk_embs,
                    dtype=tf.float32
                )
            outputs = tf.concat(gru_outs, axis=-1)
        # shape of outputs = [batch_size, sequence_length, gru_cells * 2]

    return outputs
