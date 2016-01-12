from os.path import isfile
from scipy.io import loadmat, savemat
from numpy import savetxt, concatenate, hstack, array, zeros, argmax
from util import normalize, one_hot

from keras.models import Graph
from keras.layers.core import Dense, Dropout, Reshape, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils.visualize_util import plot

def train(graph, feat, num):
    nb_epoch = 5
    for g in graphs(graph, isSave=True, trainNum=num):
        for j in range(nb_epoch):
            print('Epoch: %d / %d' %(j+1, nb_epoch))
            g.fit({'x1':feat['x1_train'], 'x2':feat['x2_train'], 'x3':feat['x3_train'], 'y':feat['y_train']},
                validation_data={'x1':feat['x1_val'], 'x2':feat['x2_val'], 'x3':feat['x3_val'], 'y':feat['y_val']}, 
                sample_weight={'y':feat['w_train'][:, 0]}, nb_epoch=1, batch_size=256, verbose=1)
            feat['yp_val'] = g.predict({'x1':feat['x1_val'], 'x2':feat['x2_val'], 'x3':feat['x3_val']}, verbose=1)['y']
            feat['YP_val'] = argmax(feat['yp_val'], axis=1)
            acc = sum(feat['YP_val'] == argmax(feat['y_val'], axis=1)).astype(float) / feat['len_train_val']
            print('val_acc: %.4f' %(acc))

def test(graph, feat):
    feat['yp_val'] = zeros([feat['len_train_val'], 0]) 
    feat['yp_test'] = zeros([feat['len_test'], 0])
    for g in graphs(graph):
        val = g.predict({'x1':feat['x1_val'], 'x2':feat['x2_val'], 'x3':feat['x3_val']}, verbose=1)['y']
        feat['yp_val'] = hstack([feat['yp_val'], val[:, 1:2]])
        test = g.predict({'x1':feat['x1_test'], 'x2':feat['x2_test'], 'x3':feat['x3_test']}, verbose=1)['y']
        feat['yp_test'] = hstack([feat['yp_test'], test[:, 1:2]])
        result(feat)

def result(feat):
    savemat('../../result/deep_learning_keras/result_val.mat', {'eid_val':feat['eid_val'] ,'yp_val': feat['yp_val']})
    savemat('../../result/deep_learning_keras/result_test.mat', {'eid_test':feat['eid_test'] ,'yp_test': feat['yp_test']})

def draw(graph):
    plot(graph, to_file='graph.png')

def path(i):
    return '/home/user/Desktop/ML_final_project/model/deep_learning_keras/graph' + str(i) + '.h5'

def save(graph, i):
    graph.save_weights(path(i), overwrite=True)

def load(graph, i):
    graph.load_weights(path(i))

def feat1():
    feat = loadmat('../../data/feat1.mat')
    perm = feat['perm'][:, 0] - 1

    (feat['y0'], feat['y']) = one_hot(feat['y'])
    for attr in ['x1', 'x2', 'x3']:
        feat[attr] = concatenate([feat[attr + '_int'], feat[attr + '_float']], axis=len(feat[attr + '_int'].shape) - 1)
        feat[attr] = normalize(feat[attr])
        del [feat[attr + '_int'], feat[attr + '_float']]
    for attr in ['eid', 'w', 'x1', 'x2', 'x3', 'y']:
        feat[attr + '_train'] = feat[attr][perm[0:feat['len_train_train']]]
        feat[attr + '_val']   = feat[attr][perm[feat['len_train_train']:feat['len_train']]]
        feat[attr + '_test']  = feat[attr][perm[feat['len_train']:feat['len']]]
        del feat[attr]
    
    return feat

def graph1():
    opt = {'border_mode':'same', 'dim_ordering':'tf'}
    drop = 0.5

    graph = Graph()
    graph.add_input('x1', input_shape=(216,))
    graph.add_input('x2', input_shape=(30, 45))
    graph.add_input('x3', input_shape=(90, 4))

    graph.add_node(Dense(2048, activation='relu'), name='dense_x1_1', input='x1')
    graph.add_node(Dropout(drop), name='drop_x1_1', input='dense_x1_1')
    graph.add_node(Dense(2048, activation='relu'), name='dense_x1_2', input='drop_x1_1')
    graph.add_node(Dropout(drop), name='drop_x1_2', input='dense_x1_2')

    graph.add_node(Reshape((30, 45, 1)), name='reshape_x2', input='x2')
    graph.add_node(Convolution2D(128, 5, 45, activation='relu', dim_ordering='tf'), name='convolution_x2_1', input='reshape_x2')
    graph.add_node(MaxPooling2D(pool_size=(2, 1), **opt), name='max_pool_x2_1', input='convolution_x2_1')
    graph.add_node(Convolution2D(32, 1, 1, activation='relu', **opt), name='inception_x2_1x1', input='max_pool_x2_1')
    graph.add_node(Convolution2D(64, 1, 1, activation='relu', **opt), name='inception_x2_3x3_reduce', input='max_pool_x2_1')
    graph.add_node(Convolution2D(64, 3, 1, activation='relu', **opt), name='inception_x2_3x3', input='inception_x2_3x3_reduce')
    graph.add_node(Convolution2D(8, 1, 1, activation='relu', **opt), name='inception_x2_5x5_reduce', input='max_pool_x2_1')
    graph.add_node(Convolution2D(16, 5, 1, activation='relu', **opt), name='inception_x2_5x5', input='inception_x2_5x5_reduce')
    graph.add_node(MaxPooling2D(pool_size=(3, 1), strides=(1, 1), **opt), name='inception_x2_pool', input='max_pool_x2_1')
    graph.add_node(Convolution2D(16, 1, 1, activation='relu', **opt), name='inception_x2_pool_proj', input='inception_x2_pool')
    graph.add_node(MaxPooling2D(pool_size=(2, 1), **opt), name='max_pool_x2_2', 
        inputs=['inception_x2_1x1', 'inception_x2_3x3', 'inception_x2_5x5', 'inception_x2_pool_proj'], concat_axis=3)
    graph.add_node(Flatten(), name='flatten_x2', input='max_pool_x2_2')

    graph.add_node(Reshape((90, 4, 1)), name='reshape_x3', input='x3')
    graph.add_node(Convolution2D(128, 7, 4, activation='relu', dim_ordering='tf'), name='convolution_x3_1', input='reshape_x3')
    graph.add_node(MaxPooling2D(pool_size=(2, 1), **opt), name='max_pool_x3_1', input='convolution_x3_1')
    graph.add_node(Convolution2D(256, 1, 1, activation='relu', **opt), name='convolution_x3_2', input='max_pool_x3_1')
    graph.add_node(Convolution2D(256, 3, 1, activation='relu', **opt), name='convolution_x3_3', input='convolution_x3_2')
    graph.add_node(MaxPooling2D(pool_size=(2, 1), **opt), name='max_pool_x3_2', input='convolution_x3_3')
    graph.add_node(Convolution2D(64, 1, 1, activation='relu', **opt), name='inception_x3_1x1', input='max_pool_x3_2')
    graph.add_node(Convolution2D(128, 1, 1, activation='relu', **opt), name='inception_x3_3x3_reduce', input='max_pool_x3_2')
    graph.add_node(Convolution2D(128, 3, 1, activation='relu', **opt), name='inception_x3_3x3', input='inception_x3_3x3_reduce')
    graph.add_node(Convolution2D(16, 1, 1, activation='relu', **opt), name='inception_x3_5x5_reduce', input='max_pool_x3_2')
    graph.add_node(Convolution2D(32, 5, 1, activation='relu', **opt), name='inception_x3_5x5', input='inception_x3_5x5_reduce')
    graph.add_node(MaxPooling2D(pool_size=(3, 1), strides=(1, 1), **opt), name='inception_x3_pool', input='max_pool_x3_2')
    graph.add_node(Convolution2D(32, 1, 1, activation='relu', **opt), name='inception_x3_pool_proj', input='inception_x3_pool')
    graph.add_node(MaxPooling2D(pool_size=(2, 1), **opt), name='max_pool_x3_3', 
        inputs=['inception_x3_1x1', 'inception_x3_3x3', 'inception_x3_5x5', 'inception_x3_pool_proj'], concat_axis=3)
    graph.add_node(Flatten(), name='flatten_x3', input='max_pool_x3_3')

    graph.add_node(Dense(2048, activation='relu'), name='dense_1', inputs=['drop_x1_2', 'flatten_x2', 'flatten_x3'])
    graph.add_node(Dropout(drop), name='drop_1', input='dense_1')
    graph.add_node(Dense(2048, activation='relu'), name='dense_2', input='drop_1')
    graph.add_node(Dropout(drop), name='drop_2', input='dense_2')
    graph.add_node(Dense(2, activation='softmax'), name='y', input='drop_2', create_output=True)

    adam = Adam(lr=1e-4)
    graph.compile(optimizer=adam, loss={'y':'categorical_crossentropy'})

    return graph

def graphs(graph, isSave=False, trainNum=0):
    i = 61
    while True:
        if isSave:
            if isfile(path(i)):
                print('[ Model: %d (Exist) ]' %(i+1))
                i += 1
                continue
            else:
                print('[ Model: %d (New) ]' %(i+1))
                yield graph
                save(graph, i)
                i += 1
                trainNum -= 1
                if trainNum == 0:
                    break
        else:
            if isfile(path(i)):
                print('[ Model: %d ]' %(i+1))
                load(graph, i)
                i += 1
                yield graph
            else:
                break

if 'feat' not in locals():
    feat = feat1()

if 'graph' not in locals():
    graph = graph1()

