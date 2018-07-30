# feature extractors functions
import tensorflow as tf


def features_extractor(inputs, mode, **kwargs):
    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    with tf.name_scope('Features_Extractor') as scope:
        with tf.name_scope('1st_Conv_32.5x5') as scope:
            conv1_out = tf.layers.conv2d(
                inputs=inputs,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                data_format='channels_last',
                activation=tf.nn.relu if not kwargs['BN'] else None,
                use_bias=not kwargs['BN'],
                trainable=kwargs['trainable'])

            if kwargs['BN']:
                conv1_normed = tf.layers.batch_normalization(
                    inputs=conv1_out,
                    axis=0,
                    fused=True,
                    training=mode == tf.estimator.ModeKeys.TRAIN)

                conv1_out = tf.nn.relu(
                    features=conv1_normed,
                    name='ReLu')

        # Pooling Layer #1
        # First max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 28, 28, 32]
        # Output Tensor Shape: [batch_size, 14, 14, 32]
        with tf.name_scope('Max_pooling_2x2') as scope:
            pool1 = tf.layers.max_pooling2d(inputs=conv1_out, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2
        # Computes 64 features using a 5x5 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 14, 14, 32]
        # Output Tensor Shape: [batch_size, 14, 14, 64]
        with tf.name_scope('2nd_Conv_64.5x5') as scope:
            conv2_out = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                data_format='channels_last',
                activation=tf.nn.relu if not kwargs['BN'] else None,
                use_bias=not kwargs['BN'],
                trainable=kwargs['trainable'])

            if kwargs['BN']:
                conv2_normed = tf.layers.batch_normalization(
                    inputs=conv2_out,
                    axis=0,
                    fused=True,
                    training=mode == tf.estimator.ModeKeys.TRAIN)

                conv2_out = tf.nn.relu(
                    features=conv2_normed,
                    name='ReLu')

        # Pooling Layer #2
        # Second max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 14, 14, 64]
        # Output Tensor Shape: [batch_size, 7, 7, 64]
        with tf.name_scope('Max_pooling_2x2') as scope:
            pool2 = tf.layers.max_pooling2d(inputs=conv2_out, pool_size=[2, 2], strides=2)

        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 7, 7, 64]
        # Output Tensor Shape: [batch_size, 7 * 7 * 64]
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

        return pool2_flat
