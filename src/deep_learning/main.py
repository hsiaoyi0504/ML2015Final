import mooc

reload(mooc)
dataset = mooc.Dataset()
net = mooc.Net(dataset)
net.train(0)

