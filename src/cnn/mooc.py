import numpy as np
import scipy.io as io
import tensorflow as tf
import util
import sys

class Dataset(object):
    def __init__(dataset):
        dataset.data = io.loadmat('../data.mat')
        for type in ['train', 'test']:
            dataset.data['size_' + type] = dataset.data['xa_' + type].shape[0]
        
        for attr in ['xa', 'xb']:
            for type in ['train', 'test']:
                dataset.data[attr + '_' + type] = util.normalize(np.log10(dataset.data[attr + '_' + type] + 1))
                # normaliz/e
        dataset.data['y0'] = np.unique(dataset.data['y_train'])
        dataset.data['y_train'] = (dataset.data['y_train'] == dataset.data['y0']).astype(float)

        dataset.permutation = np.random.permutation(dataset.data['size_train'])
        dataset.data['size_val'] = int(dataset.data['size_train'] * 0.2)
        dataset.data['size_train'] -= dataset.data['size_val']
        for attr in ['xa', 'xb', 'y']:
            dataset.data[attr + '_val'] = dataset.data[attr + '_train'][dataset.permutation[dataset.data['size_train']:]]
            dataset.data[attr + '_train'] = dataset.data[attr + '_train'][dataset.permutation[:dataset.data['size_train']]]

        dataset.batch_size = 256
        dataset.report_interval = 5000
        dataset.report_num = 0
        dataset.val_interval = 100000
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
        net.xa = tf.placeholder('float', shape = [None, 30, 24])
        net.xb = tf.placeholder('float', shape = [None, 30])
        net.y = tf.placeholder('float', shape = [None, 2])
        net.d = tf.placeholder('float')

        net.net_dnn_0()

        net.cross_entropy = - tf.reduce_sum(net.y * tf.log(net.yp))
        net.correct_prediction = tf.equal(tf.argmax(net.yp, 1), tf.argmax(net.y, 1))
        
        net.train_step = tf.train.AdamOptimizer(1e-4).minimize(net.cross_entropy)
        net.correct = tf.reduce_sum(tf.cast(net.correct_prediction, 'float'))
        net.sess = tf.Session(config = tf.ConfigProto(inter_op_parallelism_threads = 2, intra_op_parallelism_threads = 2))
        #net.sess = tf.Session()
        net.dataset = dataset
        
        net.sess.run(tf.initialize_all_variables())

    def net_dnn_0(net):
        h = util.linear_nn(net.xb, net.d, [30, 512, 512, 512])
        h = util.full_layer(h, [512, 2])
        net.yp = tf.nn.softmax(h)

    def net_cnn_0(net):
        xar = tf.reshape(net.xa, [-1, 30, 24, 1])

        ha = util.conv_layer(xar, [7, 5, 1, 32], 'VALID') # 24 x 20 x 32
        ha = tf.nn.relu6(ha) # 24 x 20 x 32
        ha = tf.nn.max_pool(ha, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME') # 12 x 10 x 32
        ha = util.conv_layer(ha, [5, 5, 32, 64], 'VALID') # 8 x 6 x 64
        ha = tf.nn.relu6(ha) # 8 x 6 x 64
        ha = tf.nn.max_pool(ha, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME') # 4 x 3 x 64
        ha = tf.reshape(ha, [-1, 4 * 3 * 64])
        
        hb = util.linear_nn(net.xb, net.d, [30, 1024])

        h = tf.matmul(ha, util.weight([4 * 3 * 64, 1024])) + tf.matmul(hb, util.weight([1024, 1024])) + util.bias([1024])
        h = util.linear_nn(h, net.d, [1024, 1024])
        h = util.full_layer(h, [1024, 2])
        net.yp = tf.nn.softmax(h)

    def net_cnn_1(net):
        xar = tf.reshape(net.xa, [-1, 30, 24, 1])

        ha = util.conv_layer(xar, [11, 24, 1, 32], 'VALID') # 20 x 1 x 32
        ha = tf.nn.relu6(ha)
        ha = tf.nn.max_pool(ha, [1, 2, 1, 1], [1, 2, 1, 1], 'SAME') # 10 x 1 x 32
        ha = util.conv_layer(ha, [5, 1, 32, 64], 'VALID') # 6 x 1 x 64
        ha = tf.nn.relu6(ha)
        ha = tf.nn.max_pool(ha, [1, 2, 1, 1], [1, 2, 1, 1], 'SAME') # 3 x 1 x 64
        ha = tf.reshape(ha, [-1, 3 * 1 * 64])
        
        hb = util.linear_nn(net.xb, net.d, [30, 512])

        h = tf.matmul(ha, util.weight([3 * 1 * 64, 512])) + tf.matmul(hb, util.weight([512, 512])) + util.bias([512])
        h = util.linear_nn(h, net.d, [512, 512])
        h = util.full_layer(h, [512, 2])
        net.yp = tf.nn.softmax(h)
    
    def net_cnn_2(net):
        xar = tf.reshape(net.xa, [-1, 30, 24, 1])

        ha = util.conv_layer(xar, [3, 24, 1, 16], 'VALID') # 28 x 1 x 16
        ha = tf.nn.max_pool(ha, [1, 2, 1, 1], [1, 2, 1, 1], 'SAME') # 14 x 1 x 16
        ha = util.conv_layer(ha, [3, 1, 16, 32], 'VALID') # 12 x 1 x 32
        ha = tf.nn.max_pool(ha, [1, 2, 1, 1], [1, 2, 1, 1], 'SAME') # 6 x 1 x 32 
        ha = util.conv_layer(ha, [3, 1, 32, 64], 'VALID') # 4 x 1 x 64
        ha = tf.reshape(ha, [-1, 4 * 1 * 64])
        
        hb = util.linear_nn(net.xb, net.d, [30, 1024])

        h = tf.matmul(ha, util.weight([4 * 1 * 64, 1024])) + tf.matmul(hb, util.weight([1024, 1024])) + util.bias([1024])
        h = util.linear_nn(h, net.d, [1024, 1024])
        h = util.full_layer(h, [1024, 2])
        net.yp = tf.nn.softmax(h)

    def train(net, count):
        phase = 'train'
        type = 'train'
        
        print('Training')
        for i in range(int(count / net.dataset.batch_size) + 1):
            batch = net.dataset.get_batch(phase, type)

            (_, err) = net.sess.run([net.train_step, net.cross_entropy], feed_dict = {net.xa: batch[0], net.xb: batch[1], net.y: batch[2], net.d: 0.5})
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
            print('Validating')
            net.dataset.data['index_' + phase + type] = 0
            correct = 0
            for i in range(int(net.dataset.data['size_' + type] / net.dataset.batch_size) + 1):
                batch = net.dataset.get_batch(phase, type)
                correct += net.sess.run(net.correct, feed_dict = {net.xa: batch[0], net.xb: batch[1], net.y: batch[2], net.d: 1.0})
                sys.stdout.write('.')
                sys.stdout.flush()
            print(' # %d (acc = %.4f) (%s)' %(net.dataset.data['size_' + type], float(correct) / net.dataset.data['size_' + type], type))

    def test(net, type):
        phase = 'test'

        yp = np.empty([net.dataset.data['size_' + type], net.dataset.data['y0'].size])
        print('Testing')
        net.dataset.data['index_' + phase + type] = 0
        for i in range(int(net.dataset.data['size_' + type] / net.dataset.batch_size) + 1):
            batch = net.dataset.get_batch(phase, type)
            yp[i * net.dataset.batch_size:i * net.dataset.batch_size + batch[0].shape[0]] = \
                net.sess.run(net.yp, feed_dict = {net.xa: batch[0], net.xb: batch[1], net.d: 1.0})
            sys.stdout.write('.')
            sys.stdout.flush()
        print(' # %d ' % net.dataset.data['size_' + type])

        return yp
