'''
import numpy as np
import mooc

dataset = mooc.Dataset()
net = mooc.Net(dataset)
net.train(2e5)

y = net.dataset.data['y_val']
y_max = np.argmax(y, 1)

yp = net.test('val')
yp_max = np.argmax(yp, 1)
'''
in_eq = net.dataset.permutation[np.argwhere(y_max != yp_max)]

