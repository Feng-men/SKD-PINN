
"""
    @Author Jiawei_Yang
    @Date 2024/1/7 19:52
"""
import numpy
import tensorflow as tf
import os

import numpy as np
import matplotlib.pyplot as plt

import time
def relative_error(pred, exact):
    # l2 error
    if type(pred) is np.ndarray:
        return np.sqrt(np.mean(np.square(pred - exact)) / np.mean(np.square(exact)))
    return tf.sqrt(tf.reduce_mean(tf.square(pred - exact)) / tf.reduce_mean(tf.square(exact)))



u_t_1_list = []
r_t_1_list = []

p_list = []
e_list = []
p_x_list = []
e_x_list = []
p_f_list = []
e_f_list = []
loss_list = []
MSE_list = []

T = 500/1.0

CUDA_VISIBLE_DEVICES=1
global_step = tf.Variable(0, trainable=False)

def KD_sigmoid(x):
    return 1.0/(1+np.exp(-x * T))

def tanh_test(x):
    return tf.tanh(x * T)

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, u0, x_f, x_t_1, layers, dx, x, truth, T):

        self.temp_1 = numpy.ones(len(x_t_1))
        self.truth = truth
        self.x = x #用来预测的
        self.lb = np.array([0.0])
        self.ub = np.array([1.0])
        self.x0 = x0
        self.u0 = u0
        self.x_f = x_f
        self.dx = dx
        self.A_T = 0.8
        self.T = T
        self.x_t_1 = x_t_1

        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)

        # tf Placeholders
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])  # 占位
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])


        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])

        self.u_t_1_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.r_t_1_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])

        self.alpha_tf = tf.placeholder(tf.float32)

        # tf Graphs
        self.u0_pred = self.neural_net(self.x0_tf, self.weights, self.biases)
        self.u_pred = self.neural_net(self.x_f_tf, self.weights, self.biases)
        self.f_pred = self.net_f(self.x_f_tf)
        self.pre = self.neural_net(self.x, self.weights, self.biases)
        self.mes = tf.sqrt(tf.reduce_mean(
            tf.square(tf.cast(tf.reshape(self.pre, [-1]) - self.truth, dtype=tf.float64))) / tf.reduce_mean(
            tf.square(self.truth)))


        # Loss
        self.loss = tf.reduce_mean(tf.square(self.u0 - self.u0_pred)) + \
                    (1 - self.alpha_tf) * tf.reduce_mean(tf.square(self.f_pred)) + \
                    self.alpha_tf * tf.reduce_mean(tf.multiply(tf.convert_to_tensor(np.array(self.temp_1,dtype='float32')) - tanh_test(self.r_t_1_tf),
                                            tf.square(self.u_pred - self.u_t_1_tf)))
        self.learning_rate = tf.train.exponential_decay(5e-4, global_step, 2000, 0.70, staircase=True)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=global_step)


        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                                         log_device_placement=True))



        init = tf.global_variables_initializer()
        self.sess.run(init)

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32,seed=1311) * xavier_stddev,
                           dtype=tf.float32)

    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = (X - self.lb) / (self.ub - self.lb)  # 初始化
        H = tf.cast(H, tf.float32)
        for l in range(0, num_layers - 2):  # l=0,1,2,3
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        # 最后一层不用激活函数，所以单独出来
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y


    def net_f(self, x):
        u_x = self.neural_net(x, self.weights, self.biases)
        s = (-0.5 * (tf.sin(np.pi * x) + 16 * tf.sin(4 * np.pi * x) + 64 * tf.sin(
            8 * np.pi * x)) * (np.pi ** 2))
        ux = tf.gradients(u_x, x)[0]
        uxx = tf.gradients(ux, x)[0]

        f = uxx - s

        return f

    def callback(self, loss):
        print('Loss:', loss)

    def train(self):

        u_t_1 = numpy.zeros(len(self.x_t_1))
        u_t_1 = np.reshape(u_t_1, (len(self.x_t_1), 1))
        r_t_1 = numpy.zeros(len(self.x_t_1))
        r_t_1 = np.reshape(r_t_1, (len(self.x_t_1), 1))
        alpha = 0.0

        flag = 1
        flag2 = 1
        flag3 = 1
        start_time = time.time()
        start_time_all = time.time()
        for it in range(self.T):

            tf_dict = {self.x0_tf: self.x0,
                       self.u0_tf: self.u0,
                       self.x_f_tf: self.x_f,
                       self.x_tf: self.x,
                       self.u_t_1_tf: u_t_1,
                       self.r_t_1_tf: r_t_1,
                       self.alpha_tf: alpha}
            if it > 1:
                alpha = self.A_T * ((it + 1) / self.T)


            self.sess.run(self.train_op, tf_dict)
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                # loss_value, pre = self.sess.run([self.loss,self.pre], tf_dict)
                loss_value = self.sess.run(self.loss, tf_dict)

                pre, mse_value = self.sess.run([self.pre, self.mes], tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f, MseLoss: %.3e' %
                      (it, loss_value, elapsed, mse_value))
                start_time = time.time()
                loss_list.append(loss_value)
                MSE_list.append(mse_value)
                # if mse_value < 0.07 and flag == 1:
                #     alltime = time.time() - start_time_all
                #     print('Loss down to 0.05 spend time: %2f' % alltime)
                #     flag = 0
                # if mse_value < 0.05 and flag2 == 1:
                #     alltime = time.time() - start_time_all
                #     print('Loss down to 0.01 spend time: %2f' % alltime)
                #     flag2 = 0
                # if mse_value < 0.03 and flag3 == 1:
                #     alltime = time.time() - start_time_all
                #     print('Loss down to 0.001 spend time: %2f' % alltime)
                #     flag3 = 0

            #Predict
            if it % 10 == 0:
                pred = self.predict(self.x)
                p_list.append(pred)

            p_c, r = self.sess.run([self.u_pred, self.f_pred], tf_dict)
            u_t_1_list.append(p_c.copy())
            r_t_1_list.append(np.square(r.copy()))

            if it > 0:
                u_t_1 = u_t_1_list[it - 1]
                r_t_1 = r_t_1_list[it - 1]


    def predict(self, x_star):

        tf_dict = {self.x_f_tf: x_star}

        u_star = self.sess.run(self.u_pred, tf_dict)

        return u_star


if __name__ == "__main__":
    noise = 0.0

    # Doman bounds
    lb = np.array([0.0, 1.0])
    T = 30000

    N_coarse = 64
    N_fine = 128
    dx_coarse = 1 / N_coarse
    dx_fine = 1 / N_fine
    layers = [1, 64, 64, 64, 64, 64, 64, 1]
    N_f = 64
    p0 = np.zeros(N_coarse + 1)
    truth = np.array(
        [(np.sin(np.pi * i * dx_fine) + np.sin(4 * np.pi * i * dx_fine) + np.sin(8 * np.pi * i * dx_fine)) / (2) for i in
         range(0, N_fine + 1)])

    x = np.linspace(0,1,N_fine+1)
    x = np.array(x)
    x = np.reshape(x, (N_fine + 1, 1))

    x0 = np.array([0.0])
    u0 = np.array([0.0])
    x1 = np.array([1.0])
    u1 = np.array([0.0])
    xb = np.array([[0.0],
          [1.0]])
    ub = np.array([[0.0],
          [0.0]])
    print(xb.shape)
    print(x0.shape)
    X_f = np.linspace(0, 1, N_coarse + 1)
    X_f = np.array(X_f)
    X_f = np.reshape(X_f, (N_coarse + 1, 1))
    model = PhysicsInformedNN(xb, ub, X_f, X_f, layers, dx_coarse, x, truth, T)

    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    y = model.predict(x)
    fig_2 = plt.figure(1)
    ax = fig_2.add_subplot(1, 1, 1)
    ax.plot(x, y, x, truth)
    plt.show()

    for i in range(len(p_list)):
        p = np.reshape(p_list[i], (-1))
        mse = np.mean((truth - p) ** 2)
        rmse = np.sqrt(mse)
        MSE_list.append(rmse)

    truth = truth.reshape((-1, 1))
    l2 = relative_error(y, truth)
    print("l2: %.4e" % (l2))

    np.savetxt("skd_pred.csv", MSE_list, delimiter=",")
    fig_2 = plt.figure(1)
    ax = fig_2.add_subplot(1, 1, 1)
    ax.plot(loss_list, 'g', label='$\mathcal{L}_{loss}$')
    ax.plot(MSE_list, 'b', label='$\mathcal{L}_{rmse}$')
    ax.set_yscale('log')
    ax.set_xlabel('iterations')
    ax.set_ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.grid(linestyle=":")
    plt.show()