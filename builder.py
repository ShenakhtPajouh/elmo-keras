import tensorflow as tf
import keras_model
import keras_elmo
import model
import elmo


def builder(options_file, weight_file, use_character_inputs=True, embedding_weight_file=None, max_batch_size=128,
            name=None, session=None):
    graph = tf.Graph()
    with graph.as_default():
        context_character_ids = tf.placeholder('int32', shape=(None, None, 50))
        bilm = model.BidirectionalLanguageModel(options_file, weight_file, use_character_inputs=use_character_inputs,
                                                embedding_weight_file=embedding_weight_file,
                                                max_batch_size=max_batch_size)
        context_embeddings_op = bilm(context_character_ids)
        elmo_context_input = elmo.weight_layers('input', context_embeddings_op, l2_coef=0.0)

    conf = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(graph=graph, config=conf)

    with graph.as_default():
        sess.run(tf.global_variables_initializer())
        gb = tf.global_variables()
        official_ELMo_varaibles = sess.run(gb)

    def _f():
        keras_context_character_ids = tf.placeholder('int32', shape=(None, None, 50))
        keras_bilm = keras_model.BidirectionalLanguageModel(options_file, weight_file,
                                                            use_character_inputs=use_character_inputs,
                                                            embedding_weight_file=embedding_weight_file,
                                                            max_batch_size=max_batch_size, name=name)
        keras_w = keras_elmo.WeightLayer(name='input', l2_coef=0.0)
        keras_context_embeddings_op = keras_bilm(keras_context_character_ids)
        keras_elmo_context_input = keras_w(keras_context_embeddings_op['lm_embeddings'],
                                           keras_context_embeddings_op['mask'])
        assigns = []
        variables = []
        variables.extend(set(keras_bilm.variables))
        variables.extend(set(keras_w.variables))
        transformer_variables = sorted(zip((var.name.lower() for var in variables), variables), key=lambda t: t[0])
        off_ELMo_pairs = sorted(zip((var.name.lower() for var in gb), official_ELMo_varaibles), key=lambda t: t[0])
        for i in range(len(transformer_variables)):
            if transformer_variables[i][0][-12:-4] == "variable" and int(transformer_variables[i][0][-3]) % 2 == 1:
                transformer_variables[i], transformer_variables[i - 1] = transformer_variables[i - 1], \
                                                                         transformer_variables[i]
        for i in range(len(transformer_variables)):
            assigns.append(tf.assign(transformer_variables[i][1], off_ELMo_pairs[i][1]))
        return keras_elmo_context_input, assigns

    if tf.executing_eagerly() and session is None:
        node, _ = _f()
    else:
        if session is None:
            session = tf.get_default_session()
        with session.graph.as_default():
            node, assigns = _f()
        _ = session.run(assigns)

    return node
