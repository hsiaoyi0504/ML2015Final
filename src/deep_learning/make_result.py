import numpy as np
import scipy.io as io

eid = io.loadmat('../../data/data.mat')['xb_test'][:, 0]
yp = net.test('test')[:, 1]
yp_max = (yp > 0.5).astype(float)

np.savetxt('../../result/deep_learning/track_1.csv', np.vstack([eid, yp]).transpose(), fmt = '%d, %.4f')
np.savetxt('../../result/deep_learning/track_2.csv', np.vstack([eid, yp_max]).transpose(), fmt = '%d, %d')
