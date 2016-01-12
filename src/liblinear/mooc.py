from scipy.io import loadmat, savemat
from numpy import concatenate

def feat1():
    feat = loadmat('../../data/feat1.mat')
    perm = feat['perm'][:, 0] - 1

    (feat['y0'], feat['y']) = one_hot(feat['y'])
    for attr in ['x1', 'x2', 'x3']:
        feat[attr] = concatenate([feat[attr + '_int'], feat[attr + '_float']], axis=len(feat[attr + '_int'].shape) - 1)
        del [feat[attr + '_int'], feat[attr + '_float']]
    for attr in ['eid', 'w', 'x1', 'x2', 'x3', 'y']:
        feat[attr + '_train'] = feat[attr][perm[0:feat['len_train_train']]]
        feat[attr + '_val']   = feat[attr][perm[feat['len_train_train']:feat['len_train']]]
        feat[attr + '_test']  = feat[attr][perm[feat['len_train']:feat['len']]]
        del feat[attr]
    
    return feat

