import collections
import json

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import tf_logging as logging

DTYPE = 'float32'
DTYPE_INT = 'int64'
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class BidirectionalLanguageModel(tf.keras.models.Model):
    def __init__(self, options_file: str,
                 use_character_inputs=True,
                 embedding_weight_file=None,
                 max_batch_size=128, *args, **kwargs):
        """
        Creates the language model computational graph and loads weights

        Two options for input type:
            (1) To use character inputs (paired with Batcher)
                pass use_character_inputs=True, and ids_placeholder
                of shape (None, None, max_characters_per_token)
                to __call__
            (2) To use token ids as input (paired with TokenBatcher),
                pass use_character_inputs=False and ids_placeholder
                of shape (None, None) to __call__.
                In this case, embedding_weight_file is also required input

        options_file: location of the json formatted file with
                      LM hyperparameters
        weight_file: location of the hdf5 file with LM weights
        use_character_inputs: if True, then use character ids as input,
            otherwise use token ids
        max_batch_size: the maximum allowable batch size
        """
        super().__init__(*args, **kwargs)
        with open(options_file, 'r') as fin:
            options = json.load(fin)

        if not use_character_inputs:
            if embedding_weight_file is None:
                raise ValueError(
                    "embedding_weight_file is required input with "
                    "not use_character_inputs"
                )
        self._options = options
        self._embedding_weight_file = embedding_weight_file
        self._use_character_inputs = use_character_inputs
        self._max_batch_size = max_batch_size

        self._ops = {}
        self.lm_graph = BidirectionalLanguageModelGraph(self._options, name='bilm',
                                                        embedding_weight_file=self._embedding_weight_file,
                                                        use_character_inputs=self._use_character_inputs,
                                                        max_batch_size=self._max_batch_size)

    def call(self, inputs, training=None, mask=None):
        if inputs in self._ops:
            # have already created ops for this placeholder, just return them
            ret = self._ops[inputs]

        else:
            # need to create the graph
            token_embeddings = self.lm_graph(inputs=inputs)
            ops = self._build_ops(token_embeddings)
            self._ops[inputs] = ops
            ret = ops

        return ret

    def _build_ops(self, token_embeddings):
        with tf.control_dependencies([self.lm_graph.update_state_op]):
            # get the LM embeddings
            layers = [
                tf.concat([token_embeddings, token_embeddings], axis=2)
            ]
            n_lm_layers = len(self.lm_graph.lstm_outputs['forward'])
            for i in range(n_lm_layers):
                layers.append(
                    tf.concat(
                        [self.lm_graph.lstm_outputs['forward'][i],
                         self.lm_graph.lstm_outputs['backward'][i]],
                        axis=-1
                    )
                )

            # The layers include the BOS/EOS tokens.  Remove them
            sequence_length_wo_bos_eos = self.lm_graph.sequence_lengths - 2
            layers_without_bos_eos = []

            for layer in layers:
                layer_wo_bos_eos = layer[:, 1:, :]
                layer_wo_bos_eos = tf.reverse_sequence(
                    layer_wo_bos_eos,
                    self.lm_graph.sequence_lengths - 1,
                    seq_axis=1,
                    batch_axis=0,
                )
                layer_wo_bos_eos = layer_wo_bos_eos[:, 1:, :]
                layer_wo_bos_eos = tf.reverse_sequence(
                    layer_wo_bos_eos,
                    sequence_length_wo_bos_eos,
                    seq_axis=1,
                    batch_axis=0,
                )
                layers_without_bos_eos.append(layer_wo_bos_eos)

            # concatenate the layers
            lm_embeddings = tf.concat(
                [tf.expand_dims(t, axis=1) for t in layers_without_bos_eos],
                axis=1
            )

            # get the mask op without bos/eos.
            # tf doesn't support reversing boolean tensors, so cast
            # to int then back
            mask_wo_bos_eos = tf.cast(self.lm_graph.mask[:, 1:], 'int32')
            mask_wo_bos_eos = tf.reverse_sequence(
                mask_wo_bos_eos,
                self.lm_graph.sequence_lengths - 1,
                seq_axis=1,
                batch_axis=0,
            )

            mask_wo_bos_eos = mask_wo_bos_eos[:, 1:]
            mask_wo_bos_eos = tf.reverse_sequence(
                mask_wo_bos_eos,
                sequence_length_wo_bos_eos,
                seq_axis=1,
                batch_axis=0,
            )

            mask_wo_bos_eos = tf.cast(mask_wo_bos_eos, 'bool')

            return {
                'lm_embeddings': lm_embeddings,
                'lengths': sequence_length_wo_bos_eos,
                'token_embeddings': self.lm_graph.embedding,
                'mask': mask_wo_bos_eos,
            }


class BidirectionalLanguageModelGraph(tf.keras.models.Model):
    """
        Creates the computational graph and holds the ops necessary for runnint
        a bidirectional language model
    """

    def __init__(self, options, name=None, use_character_inputs=True,
                 embedding_weight_file=None,
                 max_batch_size=128):
        super().__init__(name=name)
        self.options = options
        self._max_batch_size = max_batch_size
        self.use_character_inputs = use_character_inputs
        self.embedding = None
        self.mask = None
        self.sequence_lengths = None

        if embedding_weight_file is not None:
            # get the vocab size
            with h5py.File(embedding_weight_file, 'r') as fin:
                # +1 for padding
                self._n_tokens_vocab = fin['embedding'].shape[0] + 1
        else:
            self._n_tokens_vocab = None
        self._build()

    def _build(self):
        if self.use_character_inputs:
            self._build_word_char_embeddings()
        else:
            self._build_word_embeddings()
        self._build_lstms()

    def _build_word_char_embeddings(self):
        """
        options contains key 'char_cnn': {
        'n_characters': 262,
        # includes the start / end characters
        'max_characters_per_token': 50,
        'filters': [
            [1, 32],
            [2, 32],
            [3, 64],
            [4, 128],
            [5, 256],
            [6, 512],
            [7, 512]
        ],
        'activation': 'tanh',
        # for the character embedding
        'embedding': {'dim': 16}
        # for highway layers
        # if omitted, then no highway layers
        'n_highway': 2,
        }
        """
        self.projection_dim = self.options['lstm']['projection_dim']

        cnn_options = self.options['char_cnn']
        filters = cnn_options['filters']
        self.n_filters = sum(f[1] for f in filters)
        max_chars = cnn_options['max_characters_per_token']
        char_embed_dim = cnn_options['embedding']['dim']
        n_chars = cnn_options['n_characters']
        if n_chars != 262:
            raise ValueError(
                "Set n_characters=262 after training see the README.md"
            )
        activation = None
        if cnn_options['activation'] == 'tanh':
            activation = tf.nn.tanh
        elif cnn_options['activation'] == 'relu':
            activation = tf.nn.relu

        # the character embeddings
        with tf.device("/cpu:0"):
            self.Char_Embedding = EmbeddingLookup(name="a", n_tokens=n_chars, embed_dim=char_embed_dim)

        # the convolutions
        self.ConvLayer = Convolution(max_chars=max_chars, activation=activation, filters=filters,
                                     cnn_options=cnn_options, char_embed_dim=char_embed_dim,
                                     name='CNN',
                                     trainable=self.trainable)

        # for highway and projecti  n layers
        self.n_highway = cnn_options.get('n_highway')
        self.use_highway = self.n_highway is not None and self.n_highway > 0
        self.use_proj = self.n_filters != self.projection_dim

        # set up weights for projection

        if self.use_proj:
            tf.assert_greater(self.n_filters, self.projection_dim)
            self.projection_layer = Projection(n_filters=self.n_filters, projection_dim=self.projection_dim,
                                               name="CNN_proj")

        if self.use_highway:
            highway_dim = self.n_filters

            self.transformation_layers = []

            for i in range(self.n_highway):
                self.transformation_layers.append(Transformation(highway_dim=highway_dim, name='CNN_high_%s' % i))

    def _build_word_embeddings(self):
        projection_dim = self.options['lstm']['projection_dim']

        # the word embeddings
        with tf.device("/cpu:0"):
            self.EmbeddingLookup = EmbeddingLookup(
                name="embedding", n_tokens=self._n_tokens_vocab, embed_dim=projection_dim,
                dtype=DTYPE)

    def _build_lstms(self):
        # now the LSTMs
        # these will collect the initial states for the forward
        #   (and reverse LSTMs if we are doing bidirectional)

        # parse the options
        lstm_dim = self.options['lstm']['dim']
        projection_dim = self.options['lstm']['projection_dim']
        self.n_lstm_layers = self.options['lstm'].get('n_layers', 1)
        use_skip_connections = self.options['lstm']['use_skip_connections']
        if use_skip_connections:
            print("USING SKIP CONNECTIONS")
        else:
            print("NOT USING SKIP CONNECTIONS")

        # for each direction, we'll store tensors for each layer
        self.lstm_outputs = {'forward': [], 'backward': []}
        self.lstm_state_sizes = {'forward': [], 'backward': []}
        self.lstm_init_states = {'forward': [], 'backward': []}
        self.lstm_final_states = {'forward': [], 'backward': []}
        self.init_states = {'forward': [], 'backward': []}
        self.RNN = {'forward': [], 'backward': []}
        cell_clip = self.options['lstm'].get('cell_clip')
        proj_clip = self.options['lstm'].get('proj_clip')

        self.update_state_op = None

        for direction in ['forward', 'backward']:
            for i in range(self.n_lstm_layers):
                residual = use_skip_connections and i != 0
                if projection_dim < lstm_dim:
                    # are projecting down output
                    self.lstm_cell = TFLSTMCell(
                        lstm_dim, num_proj=projection_dim, residual_connection=residual,
                        cell_clip=cell_clip, proj_clip=proj_clip)
                else:
                    self.lstm_cell = TFLSTMCell(
                        lstm_dim,
                        cell_clip=cell_clip, proj_clip=proj_clip, residual_connection=residual)

                # collect the input state, run the dynamic rnn, collect
                # the output
                # the LSTMs are stateful.  To support multiple batch sizes,
                # we'll allocate size for states up to max_batch_size,
                # then use the first batch_size entries for each batch

                if direction == 'forward':
                    i_direction = 0
                else:
                    i_direction = 1
                self.RNN[direction].append(
                    tf.keras.layers.RNN(name='RNN_{0}/RNN/MultiRNNCell/Cell{1}'.format(i_direction, i),
                                        cell=self.lstm_cell, return_sequences=True, return_state=True))

    def build(self, input_shape):
        for direction in ['forward', 'backward']:
            for i in range(self.n_lstm_layers):
                self.init_states[direction].append([
                    self.add_weight(name='Variable',
                                    shape=[self._max_batch_size, dim], trainable=False,
                                    initializer=tf.zeros_initializer(), dtype=DTYPE
                                    )
                    for dim in self.lstm_cell.state_size
                ])

    def call(self, inputs, training=None, mask=None):
        embedding = None
        if self.use_character_inputs:
            with tf.device("/cpu:0"):
                char_embedding = self.Char_Embedding(inputs)
            embedding = self.ConvLayer(char_embedding)
            batch_size_n_tokens = None

            if self.use_highway or self.use_proj:
                #   reshape from (batch_size, n_tokens, dim) to (-1, dim)
                batch_size_n_tokens = tf.shape(embedding)[0:2]
                embedding = tf.reshape(embedding, [-1, self.n_filters])

            if self.use_highway:
                for i in range(self.n_highway):
                    embedding = self.transformation_layers[i](embedding)

            # finally project down if needed
            if self.use_proj:
                embedding = self.projection_layer(embedding)

            if self.use_highway or self.use_proj:
                shp = tf.concat([batch_size_n_tokens, [self.projection_dim]], axis=0)
                embedding = tf.reshape(embedding, shp)

            self.embedding = embedding
        else:
            with tf.device("/cpu:0"):
                self.embedding = self.EmbeddingLookup(inputs)

        # the sequence lengths from input mask
        if self.use_character_inputs:
            mask = tf.reduce_any(inputs > 0, axis=2)
        else:
            mask = inputs > 0
        sequence_lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
        batch_size = tf.shape(sequence_lengths)[0]
        update_ops = []
        for direction in ['forward', 'backward']:
            if direction == 'forward':
                layer_input = self.embedding
            else:
                layer_input = tf.reverse_sequence(
                    self.embedding,
                    sequence_lengths,
                    seq_axis=1,
                    batch_axis=0
                )
            for i in range(self.n_lstm_layers):
                batch_init_states = [
                    state[:batch_size, :] for state in self.init_states[direction][i]
                ]
                hidden_seq, final_hidden_state, final_cell_state = self.RNN[direction][i](layer_input,
                                                                                          initial_state=batch_init_states)

                self.lstm_state_sizes[direction].append(self.lstm_cell.state_size)
                self.lstm_init_states[direction].append(self.init_states[direction][i])
                self.lstm_final_states[direction].append(
                    tf.nn.rnn_cell.LSTMStateTuple(final_cell_state, final_hidden_state))
                if direction == 'forward':
                    self.lstm_outputs[direction].append(hidden_seq)
                else:
                    self.lstm_outputs[direction].append(
                        tf.reverse_sequence(
                            hidden_seq,
                            sequence_lengths,
                            seq_axis=1,
                            batch_axis=0
                        )
                    )
                final_state = [final_hidden_state, final_cell_state]
                with tf.control_dependencies([hidden_seq]):
                    # update the initial states
                    for j in range(2):
                        new_state = tf.concat(
                            [final_state[j][:batch_size, :],
                             self.init_states[direction][i][j][batch_size:, :]], axis=0)
                        state_update_op = tf.assign(self.init_states[direction][i][j], new_state)
                        update_ops.append(state_update_op)
                layer_input = hidden_seq
        self.mask = mask
        self.sequence_lengths = sequence_lengths
        self.update_state_op = tf.group(*update_ops)
        return embedding


class Convolution(tf.keras.layers.Layer):  # done
    def __init__(self, filters, cnn_options, char_embed_dim, max_chars, activation, name=None, trainable=True):
        super().__init__(name=name, trainable=trainable)
        self.filters = filters
        self.cnn_options = cnn_options
        self.char_embed_dim = char_embed_dim
        self.activation = activation
        self.max_chars = max_chars
        self.w = None
        self.b = None

    def build(self, input_shape):  # done
        self.w = []
        self.b = []
        for i, (width, num) in enumerate(self.filters):
            w_init = None
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
            self.w.append(self.add_weight(
                name="W_cnn_%s" % i,
                shape=[1, width, self.char_embed_dim, num],
                initializer=w_init,
                dtype=DTYPE))
            self.b.append(self.add_weight(
                name="b_cnn_%s" % i, shape=[num], dtype=DTYPE,
                initializer=tf.constant_initializer(0.0)))

    def call(self, inputs, **kwargs):
        convolutions = []
        for i, (width, num) in enumerate(self.filters):
            conv = tf.nn.conv2d(
                inputs, self.w[i],
                strides=[1, 1, 1, 1],
                padding="VALID") + self.b[i]
            # now max pool
            conv = tf.nn.max_pool(
                conv, [1, 1, self.max_chars - width + 1, 1],
                [1, 1, 1, 1], 'VALID')

            # activation
            conv = self.activation(conv)
            conv = tf.squeeze(conv, squeeze_dims=[2])
            convolutions.append(conv)

        return tf.concat(convolutions, 2)


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
        self.b_proj_cnn = self.add_weight(
            name="b_proj", shape=[self.projection_dim],
            initializer=tf.constant_initializer(0.0),
            dtype=DTYPE)


def high(x, ww_carry, bb_carry, ww_tr, bb_tr):
    carry_gate = tf.nn.sigmoid(tf.matmul(x, ww_carry) + bb_carry)
    transform_gate = tf.nn.relu(tf.matmul(x, ww_tr) + bb_tr)
    return carry_gate * transform_gate + (1.0 - carry_gate) * x


class Transformation(tf.keras.layers.Layer):
    def __init__(self, highway_dim, trainable=True, name=None, dtype=None, **kwargs):
        super().__init__(trainable, name, dtype, **kwargs)
        self.highway_dim = highway_dim
        self.W_carry = None
        self.b_carry = None
        self.W_transform = None
        self.b_transform = None

    def build(self, input_shape):
        self.W_carry = self.add_weight(
            'W_carry', [self.highway_dim, self.highway_dim],
            # glorit init
            initializer=tf.random_normal_initializer(
                mean=0.0, stddev=np.sqrt(1.0 / self.highway_dim)),
            dtype=DTYPE)
        self.b_carry = self.add_weight(
            'b_carry', [self.highway_dim],
            initializer=tf.constant_initializer(-2.0),
            dtype=DTYPE)
        self.W_transform = self.add_weight(
            'W_transform', [self.highway_dim, self.highway_dim],
            initializer=tf.random_normal_initializer(
                mean=0.0, stddev=np.sqrt(1.0 / self.highway_dim)),
            dtype=DTYPE)
        self.b_transform = self.add_weight('b_transform', [self.highway_dim],
                                           initializer=tf.constant_initializer(0.0),
                                           dtype=DTYPE)

    def call(self, inputs, **kwargs):
        return high(inputs, self.W_carry, self.b_carry,
                    self.W_transform, self.b_transform)


class EmbeddingLookup(tf.keras.layers.Layer):

    def __init__(self, n_tokens, embed_dim, trainable=True, name=None, dtype=None, **kwargs):
        super().__init__(trainable, name, dtype, **kwargs)
        self.n_chars = n_tokens
        self.char_embed_dim = embed_dim
        self.embedding_weights = None

    def build(self, input_shape):
        self.embedding_weights = self.add_weight(
            "char_embed", [self.n_chars, self.char_embed_dim],
            dtype=DTYPE,
            initializer=tf.random_uniform_initializer(-1.0, 1.0)
        )

    def call(self, inputs, **kwargs):
        return tf.nn.embedding_lookup(self.embedding_weights,
                                      inputs)


class TFLSTMCell(tf.keras.layers.Layer):
    def __init__(self, num_units,
                 use_peepholes=False, cell_clip=None,
                 initializer=None, num_proj=None, proj_clip=None,
                 num_unit_shards=None, num_proj_shards=None,
                 forget_bias=1.0, state_is_tuple=False,
                 activation=None, trainable=None, name=None, dtype=None, residual_connection=False, **kwargs):
        """Initialize the parameters for an LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell.
          use_peepholes: bool, set True to enable diagonal/peephole connections.
          cell_clip: (optional) A float value, if provided the cell state is clipped
            by this value prior to the cell output activation.
          initializer: (optional) The initializer to use for the weight and
            projection matrices.
          num_proj: (optional) int, The output dimensionality for the projection
            matrices.  If None, no projection is performed.
          proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
            provided, then the projected values are clipped elementwise to within
            `[-proj_clip, proj_clip]`.
          num_unit_shards: Deprecated, will be removed by Jan. 2017.
            Use a variable_scope partitioner instead.
          num_proj_shards: Deprecated, will be removed by Jan. 2017.
            Use a variable_scope partitioner instead.
          forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of
            the training. Must set it manually to `0.0` when restoring from
            CudnnLSTM trained checkpoints.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  This latter behavior will soon be deprecated.
          activation: Activation function of the inner states.  Default: `tanh`. It
            could also be string that is within Keras activation function names.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
          name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
          dtype: Default dtype of the layer (default of `None` means use the type
            of the first input). Required when `build` is called before `call`.
          **kwargs: Dict, keyword named properties for common layer attributes, like
            `trainable` etc when constructing the cell from configs of get_config().

          When restoring from CudnnLSTM-trained checkpoints, use
          `CudnnCompatibleLSTMCell` instead.
        """
        super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if num_unit_shards is not None or num_proj_shards is not None:
            logging.warn(
                "%s: The num_unit_shards and proj_unit_shards parameters are "
                "depreca    ted and will be removed in Jan 2017.  "
                "Use a variable scope with a partitioner instead.", self)
        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn("%s: Note that this cell is not optimized for performance. "
                         "Please use tf.contrib.cudnn_rnn.CudnnLSTM for better "
                         "performance on GPU.", self)

        # Inputs must be 2-dimensional.
        self.input_spec = input_spec.InputSpec(ndim=2)

        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializers.get(initializer)
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self.residual_connection = residual_connection
        self._kernel = None
        self._bias = None
        self._w_f_diag = None
        self._w_i_diag = None
        self._w_o_diag = None
        self._proj_kernel = None
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh

        if num_proj:
            self.state_size = [num_proj, num_units]
            self._output_size = num_proj

        else:
            self._state_size = [num_units, num_units]
            self._output_size = num_units

    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % str(inputs_shape))

        input_depth = inputs_shape[-1]
        h_depth = self._num_units if self._num_proj is None else self._num_proj
        self._kernel = self.add_weight(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + h_depth, 4 * self._num_units],
            initializer=self._initializer)
        if self.dtype is None:
            initializer = init_ops.zeros_initializer
        else:
            initializer = init_ops.zeros_initializer(dtype=self.dtype)
        self._bias = self.add_weight(
            _BIAS_VARIABLE_NAME,
            shape=[4 * self._num_units],
            initializer=initializer)
        if self._use_peepholes:
            self._w_f_diag = self.add_weight("w_f_diag", shape=[self._num_units],
                                             initializer=self._initializer)
            self._w_i_diag = self.add_weight("w_i_diag", shape=[self._num_units],
                                             initializer=self._initializer)
            self._w_o_diag = self.add_weight("w_o_diag", shape=[self._num_units],
                                             initializer=self._initializer)

        if self._num_proj is not None:
            self._proj_kernel = self.add_variable(
                "projection/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[self._num_units, self._num_proj],
                initializer=self._initializer)

        self.built = True

    def call(self, inputs, state, training=None):
        """Run one step of LSTM.

        Args:
          inputs: input Tensor, must be 2-D, `[batch, input_size]`.
          state: if `state_is_tuple` is False, this must be a state Tensor,
            `2-D, [batch, state_size]`.  If `state_is_tuple` is True, this must be a
            tuple of state Tensors, both `2-D`, with column sizes `c_state` and
            `m_state`.

        Returns:
          A tuple containing:

          - A `2-D, [batch, output_dim]`, Tensor representing the output of the
            LSTM after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of LSTM after reading `inputs` when
            the previous state was `state`.  Same type and shape(s) as `state`.

        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        num_proj = self._num_units if self._num_proj is None else self._num_proj
        sigmoid = math_ops.sigmoid

        print("State:")
        print(state)
        print("input:")
        print(inputs)
        print("proj num")
        print(self._num_proj)

        [m_prev, c_prev] = state
        input_size = inputs.get_shape().with_rank(2).dims[1].value
        if input_size is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        lstm_matrix = tf.matmul(
            array_ops.concat([inputs, m_prev], 1), self._kernel)
        lstm_matrix = nn_ops.bias_add(lstm_matrix, self._bias)

        i, j, f, o = array_ops.split(
            value=lstm_matrix, num_or_size_splits=4, axis=1)
        # Diagonal connections
        if self._use_peepholes:
            c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
                 sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
        else:
            c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
                 self._activation(j))

        if self._cell_clip is not None:
            # pylint: disable=invalid-unary-operand-type
            c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
            # pylint: enable=invalid-unary-operand-type
        if self._use_peepholes:
            m = sigmoid(o + self._w_o_diag * c) * self._activation(c)
        else:
            m = sigmoid(o) * self._activation(c)

        if self._num_proj is not None:
            m = math_ops.matmul(m, self._proj_kernel)

            if self._proj_clip is not None:
                # pylint: disable=invalid-unary-operand-type
                m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
                # pylint: enable=invalid-unary-operand-type
        o = m
        if self.residual_connection:
            o = o + inputs
        return o, [m, c]
