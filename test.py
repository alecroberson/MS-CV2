import torch
from network.model import NetworkModel
from datamanager import DataManager
from misc_tools import get_mem_size

NET_CFG = 'cfg/yolov3-mid-416.cfg' # 'cfg/squeezedet.cfg' # network configuration file
TRAIN_FILE = 'train-data.pt' # training images/labels directory
TEST_FILE = 'test-data.pt' # testing images/labels directory
AUG_CUDA = False
DATA_AUG={}

net = NetworkModel(NET_CFG, True)
net.to('cuda:0')
train_data = DataManager(
    data_path = TRAIN_FILE,
    CUDA=AUG_CUDA,
    **DATA_AUG)
test_data = DataManager(
    data_path = TEST_FILE,
    CUDA=AUG_CUDA)

optimizer = torch.optim.SGD( 
    net.parameters(), 
    lr=.001, 
    momentum=.9, 
    weight_decay=.001, 
    nesterov=False)

bs = train_data.batches(batch_size=32, mini_batch_size=32)
x, y = bs[0][0]