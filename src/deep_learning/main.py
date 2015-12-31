# import mooc0 as mooc # 0.962244 / 0.882248
import mooc1 as mooc

reload(mooc)
dataset = mooc.Dataset()
net = mooc.Net(dataset)
net.train(0)

