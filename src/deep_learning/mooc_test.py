import scipy.io as io
import tensorflow as tf
import util
import mooc

class Dataset(mooc.Dataset):
    def __init__(dataset):
        mooc.Dataset.__init__(dataset)

        dataset.data = io.loadmat('../../data/data30.mat')
        dataset.preprocess()
        dataset.partition()
        
        dataset.batch_size = 256
        dataset.batch_per_dot = 10
        dataset.report_interval = 50000
        dataset.val_interval = 100000

class Net(mooc.Net):
    def __init__(net, dataset):
        mooc.Net.__init__(net, dataset)
        
        net.xa = tf.placeholder('float', shape = [None, 30, 24])
        net.xb = tf.placeholder('float', shape = [None, 30])
        net.y = tf.placeholder('float', shape = [None, 2])
        net.d = tf.placeholder('float')
       
        h = util.full_layer(net.xb, [30, 256])
        h = util.full_layer(h, [256, 256])
        h = util.full_layer(h, [256, 2])
        net.yp = tf.nn.softmax(h)
        
        net.cross_entropy = - tf.reduce_sum(net.y * tf.log(net.yp))
        net.train_step = tf.train.AdamOptimizer(1e-4).minimize(net.cross_entropy)
        net.sess = tf.Session()
        net.sess.run(tf.initialize_all_variables())

