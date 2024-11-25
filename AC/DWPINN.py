import tensorflow as tf
import numpy as np
import scipy.io
from pyDOE import lhs

import time

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x0, u0, tb, X_star_inner, u_star_inner, X_f, layers, lb, ub):

        X0 = np.concatenate((x0, 0 * x0), 1)  # (x0, 0)
        X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
        X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)

        self.lb = lb
        self.ub = ub

        self.x0 = X0[:, 0:1]
        self.t0 = X0[:, 1:2]

        self.x_lb = X_lb[:, 0:1]
        self.t_lb = X_lb[:, 1:2]

        self.x_ub = X_ub[:, 0:1]
        self.t_ub = X_ub[:, 1:2]

        self.x_u_inner = X_star_inner[:, 0:1]
        self.t_u_inner = X_star_inner[:, 1:2]
        self.u_inner = u_star_inner

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        self.u0 = u0
        self.layers = layers
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        # tf placeholders and graph
        Config = tf.ConfigProto(allow_soft_placement=True)
        Config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=Config)

        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])

        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb.shape[1]])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]])

        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub.shape[1]])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub.shape[1]])

        self.x_u_inner_tf = tf.placeholder(tf.float32, shape=[None, self.x_u_inner.shape[1]])
        self.t_u_inner_tf = tf.placeholder(tf.float32, shape=[None, self.t_u_inner.shape[1]])
        self.u_inner_tf = tf.placeholder(tf.float32, shape=[None, self.u_inner.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.u0_pred, _ = self.net_u(self.x0_tf, self.t0_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)
        self.u_pred, self.u_x_pred = self.net_u(self.x_u_inner_tf, self.t_u_inner_tf)
        self.u_lb_pred, self.u_x_lb_pred = self.net_u(self.x_lb_tf, self.t_lb_tf)
        self.u_ub_pred, self.u_x_ub_pred = self.net_u(self.x_ub_tf, self.t_ub_tf)

        self.loss = tf.reduce_mean(tf.square(self.f_pred)) + tf.reduce_mean(
            tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred)) + \
                    tf.reduce_mean(tf.square(self.u_inner_tf - self.u_pred))

        self.optimizer_Adam = tf.train.AdamOptimizer(0.0001)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])  # 权重采用xavier初始化
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)  # bias初始化为0
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = (np.random.randn(in_dim, out_dim)) / np.sqrt(in_dim)  # xavier初始化
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def get_weight(shape, regularizer):
        w = tf.Variable()
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
        # 添加正则项
        return w

    def get_bias(shape):
        b = tf.Variable()
        return b

    def basis_function(self, X):

        X = tf.reshape(X, [-1, 2, 1])

        temp1 = (-X) * tf.exp(-X * X / 2)

        temp2 = (1 - X * X) * tf.exp(-X * X / 2)

        temp3 = (3 * X - X * X * X) * tf.exp(-X * X / 2)
        #
        temp4 = (6 * X * X - X * X * X * X - 3) * tf.exp(-X * X / 2)
        #
        temp5 = (X * X * X * X * X - 10 * X * X * X + 15 * X) * tf.exp(-X * X / 2)
        #
        temp6 = (-X * X * X * X * X * X + 15 * X * X * X * X - 45 * X * X + 15) * tf.exp(-X * X / 2)

        linshi = tf.reshape(tf.concat([temp1, temp2, temp3, temp4, temp5], 2), [-1, 10])

        return linshi

    def act(self, X):
        return tf.math.erf(X)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        # 对输入的处理很重要
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        # H=X
        H = self.act(tf.add(tf.matmul(self.basis_function(H), weights[0]), biases[0]))
        for l in range(1, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = self.act(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        u_x = tf.gradients(u, x)[0]
        return u, u_x

    def net_f(self, x, t):
        u, u_x = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x, x)[0]

        f = u_t - 0.0001 * u_xx + 5 * u * u * u - 5 * u
        return f

    def callback(self, loss):
        print('Loss:', loss)

    def train(self, nIter):
        tf_dict = {self.x0_tf: self.x0,
                   self.t0_tf: self.t0,
                   self.x_lb_tf: self.x_lb,
                   self.t_lb_tf: self.t_lb,
                   self.x_ub_tf: self.x_ub,
                   self.t_ub_tf: self.t_ub,
                   self.u0_tf: self.u0,
                   self.x_u_inner_tf: self.x_u_inner,
                   self.t_u_inner_tf: self.t_u_inner,
                   self.u_inner_tf: self.u_inner,
                   self.x_f_tf: self.x_f,
                   self.t_f_tf: self.t_f}
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time

                loss_value = self.sess.run(
                    self.loss, tf_dict)
                print('It: %d, L2 Loss: %.3e, ' \
                      'Time: %.2fs' % \
                      (it, loss_value, elapsed))
                start_time = time.time()


    def predict(self, X_star, X, T):
        tf_dict = {self.x0_tf: X.flatten()[:, None], self.t0_tf: T.flatten()[:, None]}
        u_star = self.sess.run(self.u0_pred, tf_dict)
        return u_star


if __name__ == "__main__":
    layers = [10, 60, 60, 60, 60, 60, 1]
    N_f = 10000
    data = scipy.io.loadmat('AC.mat')
    t = data['tt'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['uu'])
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.T.flatten()[:, None]

    lb = X_star.min(0)
    ub = X_star.max(0)
    idx_x = np.random.choice(x.shape[0], 100, replace=False)
    x0 = x[idx_x, :]
    u0 = Exact[idx_x, 0:1]
    idx_t = np.random.choice(t.shape[0], 100, replace=False)
    tb = t[idx_t, :]

    idx_inner = np.random.choice(X_star.shape[0], 200)
    X_star_inner = X_star[idx_inner, :]
    u_star_inner = u_star[idx_inner, :]

    X_f = lb + (ub - lb) * lhs(2, N_f)

    model = PhysicsInformedNN(x0, u0, tb, X_star_inner, u_star_inner, X_f,
                              layers, lb, ub)
    # model = PhysicsInformedNN(tb_train, xb_train,  ub_train, t_inner_train, x_inner_train, u_inner_train, X_f_train, layers, lb, ub)
    start_time = time.time()
    model.train(11000)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    # u_predict = model.predict(X_star)
    u_predict = model.predict(X_star, X_star[:, 0:1], X_star[:, 1:2])
    error_u = np.linalg.norm(u_star - u_predict, 2) / np.linalg.norm(u_star, 2)
    print('Error u: %e' % (error_u))
    error_u2 = np.linalg.norm(u_star - u_predict, ord=np.inf) / np.linalg.norm(u_star, ord=np.inf)
    print('Error u2: %e' % (error_u2))
    np.savetxt("dwpinn_pred.csv", u_predict, delimiter=",")
