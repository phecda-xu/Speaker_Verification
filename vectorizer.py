import tensorflow as tf
import os
import numpy as np
from utils import audioFeature
from configuration import get_config
from tensorflow.contrib import rnn
config = get_config()


def creat_model(fingerprint_input):
    with tf.variable_scope("lstm"):
        lstm_cells = [rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in
                      range(config.num_layer)]
        lstm = rnn.MultiRNNCell(lstm_cells)  # define lstm op and variables
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=fingerprint_input, dtype=tf.float32,
                                       time_major=True)  # for TI-VS must use dynamic rnn
        embedded = outputs[-1]  # the last ouput is the embedded d-vector
    print("embedded size: ", embedded.shape)
    return embedded


def layer_out(filepath):
    feature_array = audioFeature(filepath)
    tf.reset_default_graph()
    # draw graph
    fingerprint_input = tf.placeholder(shape=[None, feature_array.shape[0], 40], dtype=tf.float32) # enrollment batch (time x batch x n_mel)
    embedded = creat_model(fingerprint_input)
    saver = tf.train.Saver(var_list=tf.global_variables())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # load model
        print("model path :", config.model_path)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=os.path.join(config.model_path, "Check_Point"))
        ckpt_list = ckpt.all_model_checkpoint_paths
        loaded = 0
        for model in ckpt_list:
            if config.model_num == int(model[-1]):    # find ckpt file which matches configuration model number
                print("ckpt file is loaded !", model)
                loaded = 1
                saver.restore(sess, model)  # restore variables from selected ckpt file
                break
        if loaded == 0:
            raise AssertionError("ckpt file does not exist! Check config.model_num or config.model_path.")
        input = np.transpose(feature_array, axes=(1, 0, 2))
        d_array = sess.run(embedded, feed_dict={fingerprint_input: input})
    return d_array


if __name__ == "__main__":

    d_array = layer_out("BAC009S0002W0122.wav")
    print(d_array)
    print(d_array.shape)
    print(1)
