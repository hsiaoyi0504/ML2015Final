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
        dataset.report_interval = 200000
        dataset.val_interval = 1000000

class Net(mooc.Net):
    def __init__(net, dataset):
        mooc.Net.__init__(net, dataset)
        
        net.xa = tf.placeholder('float', shape = [None, 30, 24])
        net.xb = tf.placeholder('float', shape = [None, 30])
        net.y = tf.placeholder('float', shape = [None, 2])
        net.d = tf.placeholder('float')
        
        ha = tf.reshape(net.xa, [-1, 30, 24, 1])
        ha = util.conv_layer(ha, [7, 24, 1, 16], 'VALID') # 24 x 1 x 16
        ha = tf.nn.relu6(ha)
        ha = tf.nn.max_pool(ha, [1, 2, 1, 1], [1, 2, 1, 1], 'SAME') # 12 x 1 x 16
        ha = util.conv_layer(ha, [3, 1, 16, 32], 'VALID') # 10 x 1 x 32
        ha = tf.nn.relu6(ha)
        ha = tf.nn.max_pool(ha, [1, 2, 1, 1], [1, 2, 1, 1], 'SAME') # 5 x 1 x 32 
        ha = util.conv_layer(ha, [3, 1, 32, 64], 'VALID') # 3 x 1 x 64
        ha = tf.nn.relu6(ha)
        ha = tf.reshape(ha, [-1, 3 * 1 * 64])
        
        hb = util.linear_nn(net.xb, net.d, [30, 1024])

        h = tf.matmul(ha, util.weight([3 * 1 * 64, 1024])) + tf.matmul(hb, util.weight([1024, 1024])) + util.bias([1024])
        h = util.linear_nn(h, net.d, [1024, 1024, 1024])
        h = util.full_layer(h, [1024, 2])
        net.yp = tf.nn.softmax(h)
        
        net.cross_entropy = - tf.reduce_sum(net.y * tf.log(net.yp))
        net.train_step = tf.train.AdamOptimizer(1e-4).minimize(net.cross_entropy)
        net.sess = tf.Session()
        net.sess.run(tf.initialize_all_variables())
