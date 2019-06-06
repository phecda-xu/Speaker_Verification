import tensorflow as tf
import numpy as np
import os
import time
from utils import random_batch, normalize, similarity, loss_cal, optim
from configuration import get_config
from tensorflow.contrib import rnn

config = get_config()


class TISVNet(object):
    def __init__(self):
        self.fingerprint_input = tf.placeholder(shape=[None, config.N*config.M, 40],
                                                dtype=tf.float32,
                                                name="fingerprint_input")
    def creat_model(self):
        with tf.variable_scope("lstm"):
            lstm_cells = [rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in
                          range(config.num_layer)]
            lstm = rnn.MultiRNNCell(lstm_cells)  # define lstm op and variables
            outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=self.fingerprint_input, dtype=tf.float32,
                                           time_major=True)  # for TI-VS must use dynamic rnn
            embedded = outputs[-1]  # the last ouput is the embedded d-vector
        return embedded

    def layer_out(self, input, model_path):
      self.fingerprint_input = tf.placeholder(shape=[None, 1 * 1, 40],
                                              dtype=tf.float32,
                                              name="fingerprint_input")
      tf.get_default_graph()
      embedded = self.creat_model()
      saver = tf.train.Saver(var_list=tf.global_variables())
      with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, model_path)
        d_vector = sess.run(embedded, feed_dict={self.fingerprint_input: input})
      return d_vector

    def train(self):
        embedded = self.creat_model()
        lr = tf.placeholder(dtype=tf.float32, name="learning_rate")  # learning rate
        global_step = tf.Variable(0, name='global_step', trainable=False)
        w = tf.get_variable("w", initializer=np.array([10], dtype=np.float32))
        b = tf.get_variable("b", initializer=np.array([-5], dtype=np.float32))
        sim_matrix = similarity(embedded, w, b)
        loss = loss_cal(sim_matrix, type=config.loss)
        trainable_vars = tf.trainable_variables()  # get variable list
        optimizer = optim(lr)  # get optimizer (type is determined by configuration)
        grads, vars = zip(*optimizer.compute_gradients(loss))  # compute gradients of variables with respect to loss
        grads_clip, _ = tf.clip_by_global_norm(grads, 3.0)  # l2 norm clipping by 3
        grads_rescale = [0.01 * grad for grad in grads_clip[:2]] + grads_clip[2:]  # smaller gradient scale for w, b
        train_op = optimizer.apply_gradients(zip(grads_rescale, vars),
                                             global_step=global_step)  # gradient update operation
        # check variables memory
        variable_count = np.sum(np.array([np.prod(np.array(v.get_shape().as_list())) for v in trainable_vars]))
        print("total variables :", variable_count)
        tf.summary.scalar("loss", loss)
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            os.makedirs(os.path.join(config.model_path, "Check_Point"), exist_ok=True)  # make folder to save model
            os.makedirs(os.path.join(config.model_path, "logs"), exist_ok=True)  # make folder to save log
            writer = tf.summary.FileWriter(os.path.join(config.model_path, "logs"), sess.graph)
            lr_factor = 1  # lr decay factor ( 1/2 per 10000 iteration)
            loss_acc = 0  # accumulated loss ( for running average of loss)
            for iter in range(config.iteration):
                # run forward and backward propagation and update parameters
                _, loss_cur, summary = sess.run([train_op, loss, merged],
                                                feed_dict={self.fingerprint_input: random_batch(), lr: config.lr * lr_factor})
                loss_acc += loss_cur  # accumulated loss for each 100 iteration
                if iter % 10 == 0:
                    writer.add_summary(summary, iter)  # write at tensorboard
                if (iter + 1) % 100 == 0:
                    print("(iter : %d) loss: %.4f" % ((iter + 1), loss_acc / 100))
                    loss_acc = 0  # reset accumulated loss
                if (iter + 1) % 1000 == 0:
                    lr_factor /= 2  # lr decay
                    print("learning rate is decayed! current lr : ", config.lr * lr_factor)
                if (iter + 1) % 1000 == 0:
                    saver.save(sess, os.path.join(config.model_path, "./Check_Point/model.ckpt"), global_step=iter // 1000)
                    print("model is saved!")

    def test(self):
        enroll = tf.placeholder(shape=[None, config.N * config.M, 40], dtype=tf.float32,
                                name="enroll")  # enrollment batch (time x batch x n_mel)
        verif = tf.placeholder(shape=[None, config.N * config.M, 40], dtype=tf.float32,
                               name="verif")  # verification batch (time x batch x n_mel)
        self.fingerprint_input = tf.concat([enroll, verif], axis=1, name="fingerprint_input")
        embedded = self.creat_model()
        enroll_embed = normalize(
          tf.reduce_mean(tf.reshape(embedded[:config.N * config.M, :], shape=[config.N, config.M, -1]), axis=1))
        # verification embedded vectors
        verif_embed = embedded[config.N * config.M:, :]
        similarity_matrix = similarity(embedded=verif_embed, w=1., b=0., center=enroll_embed)
        saver = tf.train.Saver(var_list=tf.global_variables())
        with tf.Session() as sess:
          tf.global_variables_initializer().run()
          # load model
          print("model path :", config.model_path)
          ckpt = tf.train.get_checkpoint_state(checkpoint_dir=os.path.join(config.model_path, "Check_Point"))
          ckpt_list = ckpt.all_model_checkpoint_paths
          loaded = 0
          for model in ckpt_list:
            if config.model_num == int(model[-1]):  # find ckpt file which matches configuration model number
              print("ckpt file is loaded !", model)
              loaded = 1
              saver.restore(sess, model)  # restore variables from selected ckpt file
              break
          if loaded == 0:
            raise AssertionError("ckpt file does not exist! Check config.model_num or config.model_path.")
          print("test file path : ", config.test_path)
          # return similarity matrix after enrollment and verification
          time1 = time.time()  # for check inference time
          if config.tdsv:
            S = sess.run(similarity_matrix, feed_dict={enroll: random_batch(shuffle=False, noise_filenum=1),
                                                       verif: random_batch(shuffle=False, noise_filenum=2)})
          else:
            S = sess.run(similarity_matrix, feed_dict={enroll: random_batch(shuffle=False),
                                                       verif: random_batch(shuffle=False, utter_start=config.M)})
          S = S.reshape([config.N, config.M, -1])
          time2 = time.time()
          np.set_printoptions(precision=2)
          print("inference time for %d utterences : %0.2fs" % (2 * config.M * config.N, time2 - time1))
          print(S)  # print similarity matrix
          # calculating EER
          diff = 1
          EER = 0
          EER_thres = 0
          EER_FAR = 0
          EER_FRR = 0
          # through thresholds calculate false acceptance ratio (FAR) and false reject ratio (FRR)
          for thres in [0.01 * i + 0.5 for i in range(50)]:
            S_thres = S > thres
            # False acceptance ratio = false acceptance / mismatched population (enroll speaker != verification speaker)
            FAR = sum([np.sum(S_thres[i]) - np.sum(S_thres[i, :, i]) for i in range(config.N)]) / (
                  config.N - 1) / config.M / config.N
            # False reject ratio = false reject / matched population (enroll speaker = verification speaker)
            FRR = sum([config.M - np.sum(S_thres[i][:, i]) for i in range(config.N)]) / config.M / config.N
            # Save threshold when FAR = FRR (=EER)
            if diff > abs(FAR - FRR):
              diff = abs(FAR - FRR)
              EER = (FAR + FRR) / 2
              EER_thres = thres
              EER_FAR = FAR
              EER_FRR = FRR
          print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EER, EER_thres, EER_FAR, EER_FRR))


def main(train=True):
    net = TISVNet()
    if train:
      net.train()
    else:
      net.test()


if __name__ == "__main__":

    # main(train=config.train)
    net = TISVNet()
    input = random_batch(1,1)
    model_path = '/run/user/1001/gvfs/smb-share:server=fs.lm,share=home/xuhongyang/speaker_verification/dataSet_zy/tisv_model/./Check_Point/model.ckpt-0'
    resu = net.layer_out(input,model_path)
    print(resu)
    print(resu.shape)



