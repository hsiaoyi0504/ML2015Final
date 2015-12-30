test_time = 10

yp = np.empty([test_time, net.dataset.data['size_test', net.dataset.data['y0'].size])
for i in range(test_time):
    yp[i] = net.test('test')

yp_mean = np.mean(yp, 0)
yp_std = np.std(yp, 0)
    
