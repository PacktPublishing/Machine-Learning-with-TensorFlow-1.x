import tensorflow as tf


def compute_loss(logits, labels):
    labels = tf.squeeze(tf.cast(labels, tf.int32))

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_loss= tf.reduce_mean(cross_entropy)
    reg_loss = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    return cross_entropy_loss + reg_loss, cross_entropy_loss, reg_loss


def compute_accuracy(logits, labels):
    labels = tf.squeeze(tf.cast(labels, tf.int32))
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    predicted_correctly = tf.equal(batch_predictions, labels)
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return accuracy


def get_learning_rate(global_step, initial_value, decay_steps, decay_rate):
    learning_rate = tf.train.exponential_decay(initial_value, global_step, decay_steps, decay_rate, staircase=True)
    return learning_rate


def train(total_loss, learning_rate, global_step):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(total_loss, global_step)
    return train_op


def average_gradients(gradients):
    average_grads = []
    for grad_and_vars in zip(*gradients):
        grads = []
        for g, _ in grad_and_vars:
            grads.append(tf.expand_dims(g, 0))

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
