import tensorflow as tf
import nets, datasets
import os


def compute_loss(logits, labels):
    labels = tf.squeeze(tf.cast(labels, tf.int32))

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def compute_accuracy(logits, labels):
    labels = tf.squeeze(tf.cast(labels, tf.int32))
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    predicted_correctly = tf.equal(batch_predictions, labels)
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return accuracy


def get_learning_rate(global_step, initial_value, decay_steps, decay_rate):
    learning_rate = tf.train.exponential_decay(initial_value, global_step, decay_steps, decay_rate, staircase=True)
    return learning_rate


def train(total_loss, learning_rate, global_step, train_vars):
    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_variables = train_vars.split(",")

    grads = optimizer.compute_gradients(
        total_loss,
        [v for v in tf.trainable_variables() if v.name in train_variables]
    )
    train_op = optimizer.apply_gradients(grads, global_step=global_step)
    return train_op


def export_saved_model(sess, export_path, input_tensor, output_tensor):
    from tensorflow.python.saved_model import builder as saved_model_builder
    from tensorflow.python.saved_model import signature_constants
    from tensorflow.python.saved_model import signature_def_utils
    from tensorflow.python.saved_model import tag_constants
    from tensorflow.python.saved_model import utils
    builder = saved_model_builder.SavedModelBuilder(export_path)

    prediction_signature = signature_def_utils.build_signature_def(
        inputs={'images': utils.build_tensor_info(input_tensor)},
        outputs={
            'scores': utils.build_tensor_info(output_tensor)
        },
        method_name=signature_constants.PREDICT_METHOD_NAME)

    legacy_init_op = tf.group(
        tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            'predict_images':
                prediction_signature,
        },
        legacy_init_op=legacy_init_op)

    builder.save()


def get_model_path_from_ckpt(checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        return ckpt.model_checkpoint_path
    return None


def export_model(checkpoint_dir, export_dir, export_name, export_version):
    graph = tf.Graph()
    with graph.as_default():
        image = tf.placeholder(tf.float32, shape=[None, None, 3])
        processed_image = datasets.preprocessing(image, is_training=False)
        with tf.variable_scope("models"):
            logits = nets.inference(images=processed_image, is_training=False)

        model_checkpoint_path = get_model_path_from_ckpt(checkpoint_dir)
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.7

        with tf.Session(graph=graph) as sess:
            saver.restore(sess, model_checkpoint_path)
            export_path = os.path.join(export_dir, export_name, str(export_version))
            export_saved_model(sess, export_path, image, logits)
            print("Exported model at", export_path)


