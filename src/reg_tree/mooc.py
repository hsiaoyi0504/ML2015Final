from os.path import isfile
from scipy.io import loadmat, savemat
from numpy import concatenate, hstack, zeros
from numpy import mean, std
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals.joblib import dump, load

num_model = 16

def normalize(x):
    return (x - mean(x, 0)) / (std(x, 0) + 1e-9)

def path(i):
    return '/home/user/Desktop/ML_final_project/model/reg_tree/tree' + str(i) + '.pkl'

def base_path():
    i = 0
    while True:
        if not isfile(path(i)):
            break
        else:
            i += 1
    return i

#
def train(model, feat):
    i = base_path()

    for n in range(num_model):
        print('Model %d/%d' %(n+1, num_model))
        model.fit(feat['x1_train'], feat['y_train'])
        feat['yp_val'] = model.predict(feat['x1_val'])
        print('val_acc: %.4f' %(sum((feat['yp_val'] > 0.5) == feat['y_val']).astype(float) / feat['len_train_val'][0][0]))
        dump(model, path(i + n))

def test(model, feat):
    feat['yp_val'] = zeros([feat['len_train_val'], 0]) 
    feat['yp_test'] = zeros([feat['len_test'], 0])
    i = 0
    while isfile(path(i)):
        print('Model %d' %(i+1))
        model = load(path(i))
        val = model.predict(feat['x1_val'])
        feat['yp_val'] = hstack([feat['yp_val'], val[:, None]])
        test = model.predict(feat['x1_test'])
        feat['yp_test'] = hstack([feat['yp_test'], test[:, None]])
        i += 1

def result(feat):
    result_path = '/home/user/final/ML2015Final/result/reg_tree/' 
    savemat(result_path + 'result_val.mat', {'eid_val':feat['eid_val'] ,'yp_val': feat['yp_val']})
    savemat(result_path + 'result_test.mat', {'eid_test':feat['eid_test'] ,'yp_test': feat['yp_test']})

def feat1():
    feat = loadmat('../../data/feat1.mat')
    perm = feat['perm'][:, 0] - 1
    
    for attr in ['x1', 'x2', 'x3']:
        feat[attr] = concatenate([feat[attr + '_int'], feat[attr + '_float']], axis=len(feat[attr + '_int'].shape) - 1)
        feat[attr] = normalize(feat[attr])
        del [feat[attr + '_int'], feat[attr + '_float']]
    for attr in ['eid', 'w', 'x1', 'x2', 'x3', 'y']:
        feat[attr + '_train'] = feat[attr][perm[0:feat['len_train_train']]]
        feat[attr + '_val']   = feat[attr][perm[feat['len_train_train']:feat['len_train']]]
        feat[attr + '_test']  = feat[attr][perm[feat['len_train']:feat['len']]]
        del feat[attr]
    for attr in ['w', 'y']:
        for set in ['train', 'val', 'test']:
            feat[attr + '_' + set] = feat[attr + '_' + set][:, 0]

    return feat

def model0():
    model = RandomForestRegressor(n_estimators=128, n_jobs=8, verbose=3)

    return model

if 'feat' not in locals():
    feat = feat1()

if 'model' not in locals():
    model = model0()
