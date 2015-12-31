import numpy as np
import scipy.io as io
import tensorflow as tf
import util
import sys

class Dataset(object):
    def __init__(dataset):
        dataset.data = io.loadmat('../../data/data120.mat')
        for type in ['train', 'test']:
            dataset.data['size_' + type] = dataset.data['xa_' + type].shape[0]
        
        for attr in ['xa', 'xb']:
            for type in ['train', 'test']:
                dataset.data[attr + '_' + type] = util.normalize(np.log10(dataset.data[attr + '_' + type] + 1))
        dataset.data['y0'] = np.unique(dataset.data['y_train'])
        dataset.data['y_train'] = (dataset.data['y_train'] == dataset.data['y0']).astype(float)

        dataset.permutation = np.random.permutation(dataset.data['size_train'])
        dataset.data['size_val'] = int(dataset.data['size_train'] * 0.05)
        dataset.data['size_train'] -= dataset.data['size_val']
        for attr in ['xa', 'xb', 'y']:
            dataset.data[attr + '_val'] = dataset.data[attr + '_train'][dataset.permutation[dataset.data['size_train']:]]
            dataset.data[attr + '_train'] = dataset.data[attr + '_train'][dataset.permutation[:dataset.data['size_train']]]

        dataset.batch_size = 1024
        dataset.batch_per_dot = 8
        dataset.report_interval = 20000
        dataset.val_interval = 100000

        dataset.report_num = 0
        dataset.val_num = 0
        dataset.total_size = 0
        dataset.data['index_' + 'train' + 'train'] = 0

    def get_batch(dataset, phase, type):
        index = 'index_' + phase + type
        size = 'size_' + type

        if (dataset.data[index] == dataset.data[size]) & (phase == 'train'):
            dataset.data[index] = 0
            permutation = np.random.permutation(dataset.data[size])
            for attr in ['xa', 'xb', 'y']:
                dataset.data[attr + '_' + type] = dataset.data[attr + '_' + type][permutation]

        begin = dataset.data[index]
        dataset.data[index] = min([dataset.data[index] + dataset.batch_size, dataset.data[size]])
        end = dataset.data[index]
        if phase == 'train':
            dataset.total_size += dataset.batch_size

        if (phase == 'train') | (phase == 'val'):
            return dataset.data['xa_' + type][begin:end], dataset.data['xb_' + type][begin:end], dataset.data['y_' + type][begin:end]
        elif phase == 'test':
            return dataset.data['xa_' + type][begin:end], dataset.data['xb_' + type][begin:end]

class Net(object):
    def __init__(net, dataset):
        net.dataset = dataset

        net.net_cnn_1()

        #net.sess = tf.Session(config = tf.ConfigProto(inter_op_parallelism_threads = 2, intra_op_parallelism_threads = 2))
        net.sess = tf.Session()
        net.sess.run(tf.initialize_all_variables())

    def train(net, count):
        phase = 'train'
        type = 'train'
        
        print('Training')
        for i in range(int(count / net.dataset.batch_size) + 1):
            batch = net.dataset.get_batch(phase, type)

            (_, err) = net.sess.run([net.train_step, net.cross_entropy], feed_dict = {net.xa: batch[0], net.xb: batch[1], net.y: batch[2], net.d: 0.5})
            if i % net.dataset.batch_per_dot == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            
            if int(net.dataset.total_size / net.dataset.report_interval) != net.dataset.report_num:
                net.dataset.report_num = int(net.dataset.total_size / net.dataset.report_interval) 
                print(' # %d (ce = %.3f)' % (net.dataset.total_size, err / batch[0].shape[0]))

            if int(net.dataset.total_size / net.dataset.val_interval) != net.dataset.val_num:
                net.dataset.val_num = int(net.dataset.total_size / net.dataset.val_interval) 
                net.val()
                print('Training')

    def val(net):
        phase = 'val'

        for type in ['train', 'val']:
            yp = net.test(type)
            accuracy = sum(np.argmax(yp, 1) == np.argmax(net.dataset.data['y_' + type], 1)).astype(float) / net.dataset.data['size_' + type]
            print('# %s (acc = %.4f)' %(type, accuracy))

    def test(net, type):
        phase = 'test'

        yp = np.empty([net.dataset.data['size_' + type], net.dataset.data['y0'].size])
        print('Testing')
        net.dataset.data['index_' + phase + type] = 0
        for i in range(int(net.dataset.data['size_' + type] / net.dataset.batch_size) + 1):
            batch = net.dataset.get_batch(phase, type)
            yp[i * net.dataset.batch_size:i * net.dataset.batch_size + batch[0].shape[0]] = \
                net.sess.run(net.yp, feed_dict = {net.xa: batch[0], net.xb: batch[1], net.d: 1.0})
            if i % net.dataset.batch_per_dot == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
        print(' # %d ' % net.dataset.data['size_' + type])

        return yp
    
    def net_cnn_0(net): # data30.mat: 0.962244 / 0.882248
        net.dataset.batch_size = 1024
        net.dataset.batch_per_dot = 10
        net.dataset.report_interval = 200000
        net.dataset.val_interval = 1000000

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
    
    def net_cnn_1(net):
        net.dataset.batch_size = 1024
        net.dataset.batch_per_dot = 10
        net.dataset.report_interval = 200000
        net.dataset.val_interval = 1000000

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
    
    def net_cnn_1(net): # data120.mat
        net.dataset.batch_size = 1024
        net.dataset.batch_per_dot = 4
        net.dataset.report_interval = 20000
        net.dataset.val_interval = 100000

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
        net.train_step = tf.train.AdamOptimizer(1e-6).minimize(net.cross_entropy)
