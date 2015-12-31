import numpy as np
import tensorflow as tf
import util
import sys

class Dataset(object):
    def __init__(dataset):
        dataset.reset()

        dataset.batch_size = 256
        dataset.batch_per_dot = 8
        dataset.report_interval = 5e4
        dataset.val_interval = 1e5

    def preprocess(dataset):
        for type in ['train', 'test']:
            dataset.data['size_' + type] = dataset.data['xa_' + type].shape[0]
        
        for attr in ['xa', 'xb']:
            for type in ['train', 'test']:
                dataset.data[attr + '_' + type] = util.normalize(np.log10(dataset.data[attr + '_' + type] + 1))
        dataset.data['y0'] = np.unique(dataset.data['y_train'])
        dataset.data['y_train'] = (dataset.data['y_train'] == dataset.data['y0']).astype(float)
        dataset.data['index_' + 'train' + 'train'] = 0
        
    def partition(dataset):
        dataset.permutation = np.random.permutation(dataset.data['size_train'])
        dataset.data['size_val'] = int(dataset.data['size_train'] * 0.05)
        dataset.data['size_train'] -= dataset.data['size_val']
        for attr in ['xa', 'xb', 'y']:
            dataset.data[attr + '_val'] = dataset.data[attr + '_train'][dataset.permutation[dataset.data['size_train']:]]
            dataset.data[attr + '_train'] = dataset.data[attr + '_train'][dataset.permutation[:dataset.data['size_train']]]

    def reset(dataset):
        dataset.report_num = 0
        dataset.val_num = 0
        dataset.total_size = 0

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
