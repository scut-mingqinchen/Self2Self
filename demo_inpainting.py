import tensorflow as tf
import network.Punet
import numpy as np

import util
import cv2
import os

TF_DATA_TYPE = tf.float32
LEARNING_RATE = 1e-4
N_PREDICTION = 100
N_SAVE = 1000
N_STEP = 150000


def train(file_path, dropout_rate, mask_rate):
    print(file_path)
    tf.reset_default_graph()
    gt = util.load_np_image(file_path)
    _, w, h, c = np.shape(gt)
    model_path = file_path[0:file_path.rfind(".")] + "/" + str(mask_rate) + "/model/Self2Self/"
    os.makedirs(model_path, exist_ok=True)
    masked_img, mask = util.mask_pixel(gt, model_path, mask_rate)
    model = network.Punet.build_inpainting_unet(masked_img, mask, 1 - dropout_rate)
    loss = model['training_error']
    summay = model['summary']
    saver = model['saver']
    our_image = model['our_image']
    avg_op = model['avg_op']
    slice_avg = model['slice_avg']
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    avg_loss = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(model_path, sess.graph)
        for step in range(N_STEP):
            _, _op, loss_value, merged, o_image = sess.run([optimizer, avg_op, loss, summay, our_image])
            avg_loss += loss_value
            if (step + 1) % N_SAVE == 0:

                print("After %d training step(s)" % (step + 1),
                      "loss  is {:.9f}".format(avg_loss / N_SAVE))
                avg_loss = 0
                sum = np.float32(np.zeros(our_image.shape.as_list()))
                for j in range(N_PREDICTION):
                    o_avg, o_image = sess.run([slice_avg, our_image])
                    sum += o_image
                o_image = np.squeeze(np.uint8(np.clip(sum / N_PREDICTION, 0, 1) * 255))
                cv2.imwrite(model_path + 'Self2Self-' + str(step + 1) + '.png', o_image)
                saver.save(sess, model_path + "model.ckpt-" + str(step + 1))

            summary_writer.add_summary(merged, step)


if __name__ == '__main__':
    path = './testsets/Set11/'
    file_list = os.listdir(path)
    mask_rate = 0.9
    for file_name in file_list:
        if not os.path.isdir(path + file_name):
            train(path + file_name, 0.3, mask_rate)
