import tensorflow as tf
import numpy as np
from network.pconv_layer import PConv2D


def get_weight(shape, gain=np.sqrt(2)):
    fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)
    w = tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))
    return w


def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    return x + tf.reshape(b, [1, -1, 1, 1])


def Pconv2d_bias(x, fmaps, kernel, mask_in=None):
    assert kernel >= 1 and kernel % 2 == 1
    x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "SYMMETRIC")
    mask_in = tf.pad(mask_in, [[0, 0], [0, 0], [1, 1], [1, 1]], "CONSTANT", constant_values=1)
    conv, mask = PConv2D(fmaps, kernel, strides=1, padding='valid',
                         data_format='channels_first')([x, mask_in])
    return conv, mask


def conv2d_bias(x, fmaps, kernel, gain=np.sqrt(2)):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain)
    w = tf.cast(w, x.dtype)
    x = tf.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], "SYMMETRIC")
    return apply_bias(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID', data_format='NCHW'))


def Pmaxpool2d(x, k=2, mask_in=None):
    ksize = [1, 1, k, k]
    x = tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding='SAME', data_format='NCHW')
    mask_out = tf.nn.max_pool(mask_in, ksize=ksize, strides=ksize, padding='SAME', data_format='NCHW')
    return x, mask_out


def maxpool2d(x, k=2):
    ksize = [1, 1, k, k]
    return tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding='SAME', data_format='NCHW')


def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, factor, 1, factor])
        x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x


def conv_lr(name, x, fmaps, p=0.7):
    with tf.variable_scope(name):
        x = tf.nn.dropout(x, p)
        return tf.nn.leaky_relu(conv2d_bias(x, fmaps, 3), alpha=0.1)


def conv(name, x, fmaps, p):
    with tf.variable_scope(name):
        x = tf.nn.dropout(x, p)
        return tf.nn.sigmoid(conv2d_bias(x, fmaps, 3, gain=1.0))


def Pconv_lr(name, x, fmaps, mask_in):
    with tf.variable_scope(name):
        x_out, mask_out = Pconv2d_bias(x, fmaps, 3, mask_in=mask_in)
        return tf.nn.leaky_relu(x_out, alpha=0.1), mask_out


def partial_conv_unet(x, mask, channel=3, width=256, height=256, p=0.7, **_kwargs):
    x.set_shape([None, channel, height, width])
    mask.set_shape([None, channel, height, width])
    skips = [x]

    n = x
    n, mask = Pconv_lr('enc_conv0', n, 48, mask_in=mask)
    n, mask = Pconv_lr('enc_conv1', n, 48, mask_in=mask)
    n, mask = Pmaxpool2d(n, mask_in=mask)
    skips.append(n)

    n, mask = Pconv_lr('enc_conv2', n, 48, mask_in=mask)
    n, mask = Pmaxpool2d(n, mask_in=mask)
    skips.append(n)

    n, mask = Pconv_lr('enc_conv3', n, 48, mask_in=mask)
    n, mask = Pmaxpool2d(n, mask_in=mask)
    skips.append(n)

    n, mask = Pconv_lr('enc_conv4', n, 48, mask_in=mask)
    n, mask = Pmaxpool2d(n, mask_in=mask)
    skips.append(n)

    n, mask = Pconv_lr('enc_conv5', n, 48, mask_in=mask)
    n, mask = Pmaxpool2d(n, mask_in=mask)
    n, mask = Pconv_lr('enc_conv6', n, 48, mask_in=mask)

    # -----------------------------------------------
    n = upscale2d(n)
    n = concat(n, skips.pop())
    n = conv_lr('dec_conv5', n, 96, p=p)
    n = conv_lr('dec_conv5b', n, 96, p=p)

    n = upscale2d(n)
    n = concat(n, skips.pop())
    n = conv_lr('dec_conv4', n, 96, p=p)
    n = conv_lr('dec_conv4b', n, 96, p=p)

    n = upscale2d(n)
    n = concat(n, skips.pop())
    n = conv_lr('dec_conv3', n, 96, p=p)
    n = conv_lr('dec_conv3b', n, 96, p=p)

    n = upscale2d(n)
    n = concat(n, skips.pop())
    n = conv_lr('dec_conv2', n, 96, p=p)
    n = conv_lr('dec_conv2b', n, 96, p=p)

    n = upscale2d(n)
    n = concat(n, skips.pop())
    n = conv_lr('dec_conv1a', n, 64, p=p)
    n = conv_lr('dec_conv1b', n, 32, p=p)
    n = conv('dec_conv1', n, channel, p=p)

    return n


def concat(x, y):
    bs1, c1, h1, w1 = x.shape.as_list()
    bs2, c2, h2, w2 = y.shape.as_list()
    x = tf.image.crop_to_bounding_box(tf.transpose(x, [0, 2, 3, 1]), 0, 0, min(h1, h2), min(w1, w2))
    y = tf.image.crop_to_bounding_box(tf.transpose(y, [0, 2, 3, 1]), 0, 0, min(h1, h2), min(w1, w2))
    return tf.transpose(tf.concat([x, y], axis=3), [0, 3, 1, 2])


def build_denoising_unet(noisy, p=0.7, is_realnoisy=False):
    _, h, w, c = np.shape(noisy)
    noisy_tensor = tf.identity(noisy)
    is_flip_lr = tf.placeholder(tf.int16)
    is_flip_ud = tf.placeholder(tf.int16)
    noisy_tensor = data_arg(noisy_tensor, is_flip_lr, is_flip_ud)
    response = tf.transpose(noisy_tensor, [0, 3, 1, 2])
    mask_tensor = tf.ones_like(response)
    mask_tensor = tf.nn.dropout(mask_tensor, 0.7) * 0.7
    response = tf.multiply(mask_tensor, response)
    slice_avg = tf.get_variable('slice_avg', shape=[_, h, w, c], initializer=tf.initializers.zeros())
    if is_realnoisy:
        response = tf.squeeze(tf.random_poisson(25 * response, [1]) / 25, 0)
    response = partial_conv_unet(response, mask_tensor, channel=c, width=w, height=h, p=p)
    response = tf.transpose(response, [0, 2, 3, 1])
    mask_tensor = tf.transpose(mask_tensor, [0, 2, 3, 1])
    data_loss = mask_loss(response, noisy_tensor, 1. - mask_tensor)
    response = data_arg(response, is_flip_lr, is_flip_ud)
    avg_op = slice_avg.assign(slice_avg * 0.99 + response * 0.01)
    our_image = response

    training_error = data_loss
    tf.summary.scalar('data loss', data_loss)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=3)
    model = {
        'training_error': training_error,
        'data_loss': data_loss,
        'saver': saver,
        'summary': merged,
        'our_image': our_image,
        'is_flip_lr': is_flip_lr,
        'is_flip_ud': is_flip_ud,
        'avg_op': avg_op,
        'slice_avg': slice_avg,
    }

    return model


def build_inpainting_unet(img, mask, p=0.7):
    _, h, w, c = np.shape(img)
    img_tensor = tf.identity(img)
    mask_tensor = tf.identity(mask)
    response = tf.transpose(img_tensor, [0, 3, 1, 2])
    mask_tensor_sample = tf.transpose(mask_tensor, [0, 3, 1, 2])
    mask_tensor_sample = tf.nn.dropout(mask_tensor_sample, 0.7) * 0.7
    response = tf.multiply(mask_tensor_sample, response)
    slice_avg = tf.get_variable('slice_avg', shape=[_, h, w, c], initializer=tf.initializers.zeros())
    response = partial_conv_unet(response, mask_tensor_sample, channel=c, width=w, height=h, p=p)
    response = tf.transpose(response, [0, 2, 3, 1])
    mask_tensor_sample = tf.transpose(mask_tensor_sample, [0, 2, 3, 1])
    data_loss = mask_loss(response, img_tensor, mask_tensor - mask_tensor_sample)
    avg_op = slice_avg.assign(slice_avg * 0.99 + response * 0.01)
    our_image = img_tensor + tf.multiply(response, 1 - mask_tensor)

    training_error = data_loss
    tf.summary.scalar('data loss', data_loss)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=3)
    model = {
        'training_error': training_error,
        'data_loss': data_loss,
        'saver': saver,
        'summary': merged,
        'our_image': our_image,
        'avg_op': avg_op,
        'slice_avg': slice_avg,
    }

    return model


def mask_loss(x, labels, masks):
    cnt_nonzero = tf.to_float(tf.count_nonzero(masks))
    loss = tf.reduce_sum(tf.multiply(tf.math.pow(x - labels, 2), masks)) / cnt_nonzero
    return loss


def data_arg(x, is_flip_lr, is_flip_ud):
    x = tf.cond(is_flip_lr > 0, lambda: tf.image.flip_left_right(x), lambda: x)
    x = tf.cond(is_flip_ud > 0, lambda: tf.image.flip_up_down(x), lambda: x)
    return x
