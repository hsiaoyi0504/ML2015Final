import scipy.io as io
import tensorflow as tf
import util
import mooc

class Dataset(mooc.Dataset):
    def __init__(dataset):
        mooc.Dataset.__init__(dataset)

        dataset.data = io.loadmat('../../data/data120.mat')
        dataset.preprocess()
        dataset.partition()
        
        dataset.batch_size = 256
        dataset.batch_per_dot = 8
        dataset.report_interval = 20000
        dataset.val_interval = 100000

class Net(mooc.Net):
    def __init__(net, dataset):
        mooc.Net.__init__(net, dataset)
        
        net.xa = tf.placeholder('float', shape = [None, 120, 24])
        net.xb = tf.placeholder('float', shape = [None, 30])
        net.y = tf.placeholder('float', shape = [None, 2])
        net.d = tf.placeholder('float')

        ha = tf.reshape(net.xa, [-1, 120, 1, 24])
        ha = tf.concat(2, [ha, ha])
        ha = tf.nn.conv2d(ha, util.weight([9, 2, 24, 128]), [1, 2, 2, 1], 'SAME') + util.bias([128]) # 60 x 1 x 128
        ha = tf.nn.relu6(ha)
        ha = tf.nn.max_pool(ha, [1, 3, 1, 1], [1, 2, 1, 1], 'SAME') # 30 x 1 x 128
        ha = tf.nn.local_response_normalization(ha, depth_radius = 5, alpha = 0.0001, beta = 0.75)

        in1 = util.conv_layer(ha, [1, 1, 128, 32], 'SAME') # 30 x 1 x 32
        in1 = tf.nn.relu6(in1)

        in3 = util.conv_layer(ha, [1, 1, 128, 64], 'SAME') # 30 x 1 x 64
        in3 = tf.nn.relu6(in3)
        in3 = util.conv_layer(in3, [3, 1, 64, 64], 'SAME') # 30 x 1 x 64
        in3 = tf.nn.relu6(in3)

        in5 = util.conv_layer(ha, [1, 1, 128, 8], 'SAME') # 30 x 1 x 8
        in5 = tf.nn.relu6(in5)
        in5 = util.conv_layer(in5, [5, 1, 8, 16], 'SAME') # 30 x 1 x 16
        in5 = tf.nn.relu6(in5)

        inp = util.pool_layer(ha, [1, 3, 1, 1], 'SAME') # 30 x 1 x 128
        inp = util.conv_layer(inp, [1, 1, 128, 16], 'SAME') # 30 x 1 x 16
        inp = tf.nn.relu6(inp)

        ha = tf.concat(3, [in1, in3, in5, inp]) # 30 x 1 x 128
        ha = tf.nn.max_pool(ha, [1, 3, 1, 1], [1, 2, 1, 1], 'SAME') # 15 x 1 x 128
        ha = tf.reshape(ha, [-1, 15 * 1 * 128])
        
        hb = util.linear_nn(net.xb, net.d, [30, 1024])

        h = tf.matmul(ha, util.weight([15 * 1 * 128, 1024])) + tf.matmul(hb, util.weight([1024, 1024])) + util.bias([1024])
        h = util.linear_nn(h, net.d, [1024, 1024, 1024])
        h = util.full_layer(h, [1024, 2])
        net.yp = tf.nn.softmax(h)
        
        net.cross_entropy = - tf.reduce_sum(net.y * tf.log(net.yp))
        net.train_step = tf.train.AdamOptimizer(1e-4).minimize(net.cross_entropy)
        net.sess = tf.Session()
        net.sess.run(tf.initialize_all_variables())
