import numpy as np
import tensorflow as tf

DTYPE = 'float32'
DTYPE_INT = 'int64'


class Convolution(tf.keras.layers.Layer):  # done
    def __init__(self, filters, cnn_options, char_embed_dim, max_chars, activation, name=None, trainable=True):
        super().__init__(name=name, trainable=trainable)
        self.filters = filters
        self.cnn_options = cnn_options
        self.char_embed_dim = char_embed_dim
        self.activation = activation
        self.w = None
        self.b = None

    def build(self, input_shape):  # done
        for i, (width, num) in enumerate(self.filters):
            if self.cnn_options['activation'] == 'relu':
                # He initialization for ReLU activation
                # with char embeddings init between -1 and 1
                # w_init = tf.random_normal_initializer(
                #    mean=0.0,
                #    stddev=np.sqrt(2.0 / (width * char_embed_dim))
                # )

                # Kim et al 2015, +/- 0.05
                w_init = tf.random_uniform_initializer(
                    minval=-0.05, maxval=0.05)
            elif self.cnn_options['activation'] == 'tanh':
                # glorot init
                w_init = tf.random_normal_initializer(
                    mean=0.0,
                    stddev=np.sqrt(1.0 / (width * self.char_embed_dim))
                )
            w = self.add_weight(
                name="W_cnn_%s" % i,
                shape=[1, width, self.char_embed_dim, num],
                initializer=w_init,
                dtype=DTYPE)
            b = self.add_weight(
                name="b_cnn_%s" % i, shape=[num], dtype=DTYPE,
                initializer=tf.constant_initializer(0.0))

    def call(self, inputs, **kwargs):  # done
        convolutions = []
        for i, (width, num) in enumerate(self.filters):
            conv = tf.nn.conv2d(
                inputs, self.w,
                strides=[1, 1, 1, 1],
                padding="VALID") + self.b
            # now max pool
            conv = tf.nn.max_pool(
                conv, [1, 1, self.max_chars - width + 1, 1],
                [1, 1, 1, 1], 'VALID')

            # activation
            conv = self.activation(conv)
            conv = tf.squeeze(conv, squeeze_dims=[2])
            convolutions.append(conv)

        return tf.concat(convolutions)

    def __call__(self, inputs, **kwargs):
        super().__call__(inputs=inputs, **kwargs)


class Projection(tf.keras.layers.Layer):
    def __init__(self, n_filters, projection_dim, trainable=True, name=None, dtype=None, **kwargs):
        super().__init__(trainable, name, dtype, **kwargs)
        self.n_filters = n_filters
        self.projection_dim = projection_dim
        self.W_proj_cnn = None
        self.b_proj_cnn = None

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, self.W_proj_cnn) + self.b_proj_cnn

    def build(self, input_shape):
        self.W_proj_cnn = self.add_weight(
            name="W_proj", shape=[self.n_filters, self.projection_dim],
            initializer=tf.random_normal_initializer(
                mean=0.0, stddev=np.sqrt(1.0 / self.n_filters)),
            dtype=DTYPE)
        self.b_proj_cnn = tf.get_variable(
            name="b_proj", shape=[self.projection_dim],
            initializer=tf.constant_initializer(0.0),
            dtype=DTYPE)

    def __call__(self, inputs, **kwargs):
        super().__call__(inputs=inputs, **kwargs)
