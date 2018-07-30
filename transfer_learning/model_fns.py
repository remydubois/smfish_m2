# model func
from transfer_learning.feature_extractors import *
from transfer_learning.tools import *


def bitask_cnn(features, labels, mode, params):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    pool2_flat = features_extractor(inputs=input_layer, BN=params['BN'], mode=mode, trainable=params['trainable'])

    # Reversing gradient layer
    # In forward mode, its output is t + tf.stop_gradient(tf.identity(pool2_flat) - t) = tf.identity(pool2_flat)
    # In backward mode, its output is t + 0 = t = grad(-identity(pool2_flat)) = -grad(pool2_flat)
    with tf.name_scope('Reverse_gradients_layer') as scope:
        t = -tf.identity(pool2_flat)
        reversing_gradients_layer = t + tf.stop_gradient(tf.identity(pool2_flat) - t)

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    with tf.name_scope('Digits_MLP') as scope:
        with tf.name_scope('Dense_for_digits') as scope:
            dense_for_digits = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout_for_digits = tf.layers.dropout(
            inputs=dense_for_digits, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
        logits = tf.layers.dense(inputs=dropout_for_digits, units=10)

    with tf.name_scope('Domain_MLP') as scope:
        dense_for_domains = tf.layers.dense(inputs=reversing_gradients_layer, units=100, activation=tf.nn.relu)
        dropout_for_domains = tf.layers.dropout(
            inputs=dense_for_domains, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
        domain_logits = tf.layers.dense(inputs=dropout_for_domains, units=2)

    predictions = {}
    predictions["digits_probabilities"] = tf.nn.softmax(logits, name="digits_probabilities")  # softmax_tensor
    predictions["domains_probabilities"] = tf.nn.softmax(domain_logits, name="domains_probabilities")
    predictions["digits"] = tf.argmax(input=predictions['digits_probabilities'], axis=1)
    predictions["domains"] = tf.argmax(input=predictions['domains_probabilities'], axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    digits_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels['digits'], logits=logits)
    domain_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels['domain'], logits=domain_logits)
    total_loss = tf.add(digits_loss, domain_loss)

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    tf.summary.scalar('domain_loss', domain_loss)
    tf.summary.histogram('digits_distribution', predictions['digits'])

    summary_hook = tf.train.SummarySaverHook(
        save_steps=40,  # specified later in call to estimator i guess. overriden anyway
        summary_op=tf.summary.merge_all())

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = params['optimizer'](learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=total_loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=digits_loss, train_op=train_op, training_hooks=[summary_hook])

    # Add evaluation metrics (for EVAL mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels['digits'], predictions=predictions["digits"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=digits_loss, eval_metric_ops=eval_metric_ops)


def standard_cnn(features, labels, mode, params):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    featured = features_extractor(inputs=input_layer, mode=mode, BN=params['BN'], trainable=params['trainable'])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    with tf.name_scope('Digits_MLP') as scope:
        with tf.name_scope('Dense_for_digits') as scope:
            dense_for_digits = tf.layers.dense(inputs=featured, units=1024, activation=tf.nn.relu)
        dropout_for_digits = tf.layers.dropout(
            inputs=dense_for_digits, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
        logits = tf.layers.dense(inputs=dropout_for_digits, units=10)

    predictions = {}
    # Generate predictions (for PREDICT and EVAL mode)
    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    # `logging_hook`.
    predictions["digits_probabilities"] = tf.nn.softmax(logits, name="digits_probabilities")  # softmax_tensor
    predictions["digits"] = tf.argmax(input=predictions['digits_probabilities'], axis=1)

    # Calculate Loss (for both TRAIN and EVAL modes)
    digits_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    # atf.summary.scalar('digits_loss',digits_loss)
    tf.summary.histogram('digits_distribution', predictions['digits'])

    summary_hook = tf.train.SummarySaverHook(
        save_steps=40,  # specified later in call to estimator i guess. overriden anyway
        # output_dir=?, no need if specified in later call to tf.estimator.Estimator by model_dir
        summary_op=tf.summary.merge_all())

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = params['optimizer'](learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=digits_loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=digits_loss, train_op=train_op, training_hooks=[summary_hook])

    # Add evaluation metrics (for EVAL mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["digits"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=digits_loss, eval_metric_ops=eval_metric_ops)

    # Configure for PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
