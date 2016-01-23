from scipy.io import loadmat
from numpy import savetxt
from numpy import mean, std, unique, argmax
from numpy import vstack, hstack, split

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Merge 
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

result_root = '/home/user/final/ML2015Final/result'

def normalize(x):
    return (x - mean(x, 0)) / (std(x, 0) + 1e-9)

def one_hot(x):
    x0 = unique(x)
    return (x0, (x == x0).astype(float))

def train(model, feat, nb_epoch):
    history = []
    for epoch in range(nb_epoch):
        h = model.fit({'yp':feat['yp_val_train'], 'y':feat['y_val_train']}, 
            validation_data={'yp':feat['yp_val_val'], 'y':feat['y_val_val']}, 
            nb_epoch=1, batch_size=feat['len_train_val_train'], verbose=1)
        history.append(h)
        feat['yp_val_val_ult'] = model.predict({'yp':feat['yp_val_val']})['y']
        acc = sum(argmax(feat['yp_val_val_ult'], axis=1) == argmax(feat['y_val_val'], axis=1)).astype(float) / feat['len_train_val_val'][0][0]
        print('Epoch %d, val_acc: %.4f' %(epoch, acc))
    return history
 
def test(model, feat):
    feat['yp_test_ult'] = model.predict({'yp':feat['yp_test']})['y'][:, 1:2]
    feat['YP_test_ult'] = (feat['yp_test_ult'] > 0.5).astype(float)

def result(feat):
    result_type = '/blending_harry'
    savetxt(result_root + result_type + '/result_track_1.csv', hstack([feat['eid_test'], feat['yp_test_ult']]), fmt='%d,%.6f')
    savetxt(result_root + result_type + '/result_track_2.csv', hstack([feat['eid_test'], feat['YP_test_ult']]), fmt='%d,%d')

def feat_blend(val_train_ratio):
    result_type = ['/deep_learning_keras/0', '/deep_learning_keras/1', '/liblinear', '/reg_tree']
   
    feat = dict()
    f = loadmat('/home/user/final/ML2015Final/data/feat1.mat')
    perm = f['perm'][:, 0] - 1

    for attr in ['len', 'len_train', 'len_train_train', 'len_train_val', 'len_test']:
        feat[attr] = f[attr]
    feat['len_train_val_train'] = int(val_train_ratio * feat['len_train_val'])
    feat['len_train_val_val'] = feat['len_train_val'] - feat['len_train_val_train']

    (f['y0'], f['y']) = one_hot(f['y'])
    for attr in ['eid', 'y']:
        (feat[attr + '_val_train'], feat[attr + '_val_val']) = split(f[attr][perm[f['len_train_train']:f['len_train']]], [feat['len_train_val_train']])
        feat[attr + '_test'] = f[attr][perm[f['len_train']:f['len']]]

    for type in result_type:
        for set in ['val', 'test']:
            fileName = result_root + type + '/result_' + set
            f = loadmat(fileName)
            name = 'yp_' + set
            print('Loaded ' + fileName + ', dimension: %d' %(f[name].shape[1]))
            if name not in feat:
                feat[name] = f[name]
            else:
                feat[name] = hstack((feat[name], f[name]))

    (feat['yp_val_train'], feat['yp_val_val'], feat['yp_test']) = \
        split(normalize(vstack([feat['yp_val'], feat['yp_test']])), [feat['len_train_val_train'], feat['len_train_val']])

    feat['dim'] = feat['yp_val'].shape[1]
        
    return feat

def model_ensemble(feat, ensemble_num):
    graph = Graph()
    graph.add_input(name='yp', input_shape=(feat['dim'],))
    
    layers = []
    for i in range(ensemble_num):
        print('Building sub-net %d' %i)
        graph.add_node(Dense(128, activation='tanh'), name='dense_' + str(i) + '_1', input='yp')
        graph.add_node(Dense(2, activation='softmax'), name='softmax_' + str(i), input='dense_' + str(i) + '_1')
        layers += ['softmax_' + str(i)]
    
    if len(layers) == 1:
        graph.add_output(name='y', input=layers[0])
    else:        
        graph.add_output(name='y', inputs=layers, merge_mode='ave')

    adam = Adam(lr=1e-3)
    graph.compile(loss={'y':'categorical_crossentropy'}, optimizer=adam)

    return graph

if 'feat' not in locals():
    val_train_ratio = 0.5
    ensemble_num = 1
    feat = feat_blend(val_train_ratio)
    model = model_ensemble(feat, ensemble_num)
