import tensorflow as tf


class WeightLayer(tf.keras.layers.Layer):
    """
    Weight the layers of a biLM with trainable scalar weights to
    compute ELMo representations.

    For each output layer, this returns two ops.  The first computes
        a layer specific weighted average of the biLM layers, and
        the second the l2 regularizer loss  term.
    The regularization terms are also add to tf.GraphKeys.REGULARIZATION_LOSSES
    bilm_ops = the tensorflow ops returned to compute internal
        representations from a biLM.  This is the return value
        from BidirectionalLanguageModel(...)(ids_placeholder)
    """

    def __init__(self, l2_coef=None, trainable=True, name=None,
                 dtype=None, **kwargs):
        """

        :param l2_coef: the l2 regularization coefficient $\lambda$.
            Pass None or 0.0 for no regularization.
        :param do_layer_norm:
        :param trainable: is trainable or not
        :param name: a string prefix used for the trainable variable names
        :param dtype: dtype
        :param kwargs:
        """
        super().__init__(trainable, name, dtype, **kwargs)
        self.l2_coef = l2_coef
        self.W = None
        self.gamma = None

    def _l2_regularizer(self, weights):
        if self.l2_coef is not None:
            return self.l2_coef * tf.reduce_sum(tf.square(weights))
        else:
            return 0.0

    def build(self, input_shape):
        self.W = self.add_weight(name='{}_ELMo_W'.format(self.name), initializer=tf.zeros_initializer,
                                 regularizer=self._l2_regularizer, trainable=True, shape=(int(input_shape[1]),))
        # scale the weighted sum by gamma
        self.gamma = self.add_weight(name='{}_ELMo_gamma'.format(self.name), shape=(1,),
                                     initializer=tf.ones_initializer, regularizer=None, trainable=True)

    def call(self, inputs, mask=None, use_top_only=False, do_layer_norm=False):
        """

        :param inputs: bilm_ops['lm_embeddings']
        :param mask: bilm_ops['mask']
        :param use_top_only: if True, then only use the top layer.
        :param do_layer_norm: if True, then apply layer normalization to each biLM
            layer before normalizing
        :return:
            {
            'weighted_op': op to compute weighted average for output,
            'regularization_op': op to compute regularization term
            }
        """
        # Get ops for computing LM embeddings and mask

        n_lm_layers = int(inputs.get_shape()[1])
        lm_dim = int(inputs.get_shape()[3])

        with tf.control_dependencies([inputs, mask]):
            # Cast the mask and broadcast for layer use.
            mask_float = tf.cast(mask, 'float32')
            broadcast_mask = tf.expand_dims(mask_float, axis=-1)

            def _do_ln(x):
                # do layer normalization excluding the mask
                x_masked = x * broadcast_mask
                N = tf.reduce_sum(mask_float) * lm_dim
                mean = tf.reduce_sum(x_masked) / N
                variance = tf.reduce_sum(((x_masked - mean) * broadcast_mask) ** 2
                                         ) / N
                return tf.nn.batch_normalization(
                    x, mean, variance, None, None, 1E-12
                )

            if use_top_only:
                layers = tf.split(inputs, n_lm_layers, axis=1)
                # just the top layer
                sum_pieces = tf.squeeze(layers[-1], squeeze_dims=1)
                # no regularization

            else:
                # normalize the weights
                normed_weights = tf.split(
                    tf.nn.softmax(self.W + 1.0 / n_lm_layers), n_lm_layers
                )
                # split LM layers
                layers = tf.split(inputs, n_lm_layers, axis=1)

                # compute the weighted, normalized LM activations
                pieces = []
                for w, t in zip(normed_weights, layers):
                    if do_layer_norm:
                        pieces.append(w * _do_ln(tf.squeeze(t, squeeze_dims=1)))
                    else:
                        pieces.append(w * tf.squeeze(t, squeeze_dims=1))
                sum_pieces = tf.add_n(pieces)

            weighted_lm_layers = sum_pieces * self.gamma

            ret = weighted_lm_layers

        return ret
