from os.path import isfile
from scipy.io import loadmat, savemat
from numpy import savetxt, concatenate, vstack, array, zeros, argmax
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
            acc = sum(feat['YP_val'] == argmax(feat['y_val'], axis=1)).astype(float) / feat['len_val']
            print('val_acc: %.4f' %(acc))

def test(graph, feat):
    feat['yp_test'] = zeros([0, feat['len_test'], 1])
    for g in graphs(graph):
        temp = g.predict({'x1':feat['x1_test'], 'x2':feat['x2_test'], 'x3':feat['x3_test']}, verbose=1)['y']
        feat['yp_test'] = vstack([feat['yp_test'], temp[None, :, 1:2]])

def result(feat):
    savemat('../../result/deep_learning_keras/result_1.mat', {'eid_test':feat['eid_test'] ,'yp_test': feat['yp_test']})
    #savetxt('../../result/deep_learning_keras/result_1_track_1.csv', vstack([feat['eid_test'][:, 0], feat['yp_test'][:, 1]]).T, fmt='%d,%.6f')
    #savetxt('../../result/deep_learning_keras/result_1_track_2.csv', vstack([feat['eid_test'][:, 0], feat['YP_test']]).T, fmt='%d,%d')

def draw(graph):
    plot(graph, to_file='graph.png')

def save(graph, i):
    name = '/home/user/Desktop/ML_final_project/graph/graph' + str(i) + '.h5'
    graph.save_weights(name, overwrite=True)

def check(graph, i):
    name = '/home/user/Desktop/ML_final_project/graph/graph' + str(i) + '.h5'
    return isfile(name)

def load(graph, i):
    name = '/home/user/Desktop/ML_final_project/graph/graph' + str(i) + '.h5'
    graph.load_weights(name)

def feat1():
    feat = loadmat('../../data/feat1.mat')
    print feat['perm'][0:feat['len_train_train']]
    print feat['perm'][feat['len_train_train']:feat['len_train']]
    print feat['perm'][feat['len_train']:feat['len']]

    (feat['y0'], feat['y']) = one_hot(feat['y'])
    for attr in ['x1', 'x2', 'x3']:
        feat[attr] = normalize(concatenate([feat[attr + '_int'], feat[attr + '_float']], axis=len(feat[attr + '_int'].shape) - 1))
        del [feat[attr + '_int'], feat[attr + '_float']]
    for attr in ['eid', 'w', 'x1', 'x2', 'x3', 'y']:
        print feat[attr].shape
        feat[attr + '_train'] = feat[attr][feat['perm'][0:feat['len_train_train']]]
        feat[attr + '_val']   = feat[attr][feat['perm'][feat['len_train_train']:feat['len_train']]]
        feat[attr + '_test']  = feat[attr][feat['perm'][feat['len_train']:feat['len']]]
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
    i = 0
    while True:
        if isSave:
            if check(graph, i):
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
            if check(graph, i):
                print('[ Model: %d ]' %(i+1))
                load(graph, i)
                i += 1
                yield graph
                break

if 'feat' not in locals():
    feat = feat1()

if 'graph' not in locals():
    graph = graph1()

train(graph, feat, 1)
