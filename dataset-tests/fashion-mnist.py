local fashion_mnist = require 'fashion-mnist'

local trainset = fashion_mnist.traindataset()
local testset = fashion_mnist.testdataset()

print(trainset.size) -- to retrieve the size
print(testset.size) -- to retrieve the size