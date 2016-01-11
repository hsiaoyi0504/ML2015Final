from scipy.io import loadmat
from numpy import genfromtxt, savetxt, concatenate, vstack, split, array, argmax
from util import normalize, one_hot, shuffle

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Reshape, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam

def train(model, feat, nb_epoch):
    model.fit([feat['x1_train'], feat['x2_train'], feat['x3_train']], feat['y_train'], 
        validation_data=([feat['x1_val'], feat['x2_val'], feat['x3_val']], feat['y_val']),
        nb_epoch=nb_epoch, batch_size=256, verbose=1, show_accuracy=True)

def test(model, feat):
    print model.evaluate([feat['x1_test'], feat['x2_test'], feat['x3_test']], feat['y_test'], 
        verbose=1, show_accuracy=True)

def result(model, feat):
    feat['yp_test'] = model.predict([feat['x1_test'], feat['x2_test'], feat['x3_test']])
    feat['YP_test'] = argmax(feat['yp_test'], axis=1)
    
    savetxt('../../result/deep_learning_keras/result_0_track_1.csv', vstack([feat['eid_test'][:, 0], feat['yp_test'][:, 1]]).T, fmt='%d,%.6f')
    savetxt('../../result/deep_learning_keras/result_0_track_2.csv', vstack([feat['eid_test'][:, 0], feat['YP_test']]).T, fmt='%d,%d')

def save(model):
    json_string = model.to_json()
    open('/home/user/Desktop/ML_final_project/model.json', 'w').write(json_string)
    model.save_weights('/home/user/Desktop/ML_final_project/model.h5', overwrite=True)

def load():
    model = model_from_json(open('/home/user/Desktop/ML_final_project/model.json').read())
    model.load_weights('/home/user/Desktop/ML_final_project/model.h5')

    return model

def feat1():
    feat = loadmat('../../data/feat1.mat')
    res0 = genfromtxt('../../result/deep_learning/result_0_track_2.csv', delimiter=',')
    feat['len_val'] = int(0.05 * feat['len_train']);

    (feat['y0'], feat['y']) = one_hot(feat['y'])
    for attr in ['x1', 'x2', 'x3']:
        feat[attr] = normalize(concatenate([feat[attr + '_int'], feat[attr + '_float']], axis=len(feat[attr + '_int'].shape) - 1))
        del [feat[attr + '_int'], feat[attr + '_float']]
    for attr in ['eid', 'w', 'x1', 'x2', 'x3', 'y']:
        (feat[attr + '_train'], feat[attr + '_test']) = split(feat[attr], array([feat['len_train']]))
        del feat[attr]
    feat['y_test'] = (res0[:, 1:2] == feat['y0']).astype(float)
    (feat['w_train'], feat['x1_train'], feat['x2_train'], feat['x3_train'], feat['y_train']) = \
        shuffle(feat['w_train'], feat['x1_train'], feat['x2_train'], feat['x3_train'], feat['y_train'])
    for attr in ['w', 'x1', 'x2', 'x3', 'y']:
        (feat[attr + '_train'], feat[attr + '_val']) = split(feat[attr + '_train'], array([feat['len_train'] - feat['len_val']]))

    return feat

def model0():
    models = [Sequential() for i in range(3)]

    models[0].add(Dense(2048, activation='relu', input_shape=(216,)))
    models[0].add(Dropout(0.25))

    models[1].add(Reshape((30, 45, 1), input_shape=(30, 45)))
    models[1].add(Convolution2D(16, 3, 45, activation='relu', dim_ordering='tf'))
    models[1].add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1), dim_ordering='tf'))
    models[1].add(Convolution2D(32, 3, 1, activation='relu', dim_ordering='tf'))
    models[1].add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1), dim_ordering='tf'))
    models[1].add(Convolution2D(64, 3, 1, activation='relu', dim_ordering='tf'))
    models[1].add(Flatten())

    models[2].add(Reshape((90, 4, 1), input_shape=(90, 4)))
    models[2].add(Convolution2D(16, 3, 4, activation='relu', dim_ordering='tf'))
    models[2].add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1), dim_ordering='tf'))
    models[2].add(Convolution2D(32, 3, 1, activation='relu', dim_ordering='tf'))
    models[2].add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1), dim_ordering='tf'))
    models[2].add(Convolution2D(64, 3, 1, activation='relu', dim_ordering='tf'))
    models[2].add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1), dim_ordering='tf'))
    models[2].add(Flatten())

    model = Sequential()
    model.add(Merge(models, mode='concat'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))

    adam = Adam(lr=1e-4)
    model.compile(optimizer=adam, loss='categorical_crossentropy')

    return model

if 'feat' not in locals():
    feat = feat1()
if 'model' not in locals():
    model = model0()
