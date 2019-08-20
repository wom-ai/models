import tensorflow as tf


@tf.contrib.framework.add_arg_scope
def depthwise_conv(
        x, kernel=3, stride=1, padding='SAME',
        activation_fn=None, normalizer_fn=None,
        data_format='NHWC', scope='depthwise_conv'):
    with tf.variable_scope(scope):

        if data_format == 'NHWC':
            in_channels = x.shape[3].value
            strides = [1, stride, stride, 1]
        else:
            in_channels = x.shape[1].value
            strides = [1, 1, stride, stride]

        W = tf.get_variable(
            'depthwise_weights',
            [kernel, kernel, in_channels, 1],
            dtype=tf.float32
        )
        x = tf.nn.depthwise_conv2d(x, W, strides, padding, data_format=data_format)
        x = normalizer_fn(x) if normalizer_fn is not None else x  # batch normalization
        x = activation_fn(x) if activation_fn is not None else x  # nonlinearity
        return x
