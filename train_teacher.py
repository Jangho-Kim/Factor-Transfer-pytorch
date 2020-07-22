
import argparse
import json
import time
from datetime import datetime
import warnings
import os
warnings.filterwarnings("ignore")

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import random
from logger import SummaryLogger
import utils
from Models import *


parser = argparse.ArgumentParser(description='Factor transfer for CIFAR10')
parser.add_argument('--text', default='log.txt', type=str)
parser.add_argument('--exp_name', default='cifar10/t_res56', type=str)
parser.add_argument('--log_time', default='1', type=str)
parser.add_argument('--lr', default='0.1', type=float)
parser.add_argument('--resume_epoch', default='0', type=int)
parser.add_argument('--epoch', default='163', type=int)
parser.add_argument('--decay_epoch', default=[82, 123], nargs="*", type=int)
parser.add_argument('--w_decay', default='5e-4', type=float)
parser.add_argument('--cu_num', default='0', type=str)
parser.add_argument('--seed', default='1', type=str)
parser.add_argument('--load_pretrained', default='  ', type=str)
parser.add_argument('--save_model', default='ckpt.t7', type=str)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = parser.parse_args()
print(args)


#### random Seed #####
num = random.randint(1, 10000)
random.seed(num)
torch.manual_seed(num)
#####################


os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num


#Data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)



#Other parameters
DEVICE = torch.device("cuda")
RESUME_EPOCH = args.resume_epoch
DECAY_EPOCH = args.decay_epoch
DECAY_EPOCH = [ep - RESUME_EPOCH for ep in DECAY_EPOCH]
FINAL_EPOCH = args.epoch
EXPERIMENT_NAME = args.exp_name
W_DECAY = args.w_decay
base_lr = args.lr

# Model
model = ResNet56()
#model = ResNet20()


model.to(DEVICE)

# Loss and Optimizer
optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=W_DECAY)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
criterion_CE = nn.CrossEntropyLoss()


def eval(net):
    loader = testloader
    flag = 'Test'

    epoch_start_time = time.time()
    net.eval()
    val_loss = 0

    correct = 0


    total = 0
    criterion_CE = nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs= net(inputs)


        loss = criterion_CE(outputs[3], targets)
        val_loss += loss.item()

        _, predicted = torch.max(outputs[3].data, 1)


        total += targets.size(0)

        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    print('%s \t Time Taken: %.2f sec' % (flag, time.time() - epoch_start_time))
    print('Loss: %.3f | Acc net: %.3f%%' % (train_loss / (b_idx + 1), 100. * correct / total))
    return val_loss / (b_idx + 1),  correct / total

def train(model, epoch):
    epoch_start_time = time.time()
    print('\n EPOCH: %d' % epoch)
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    global optimizer



    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()


        ###################################################################################
        outputs= model(inputs)
        loss = criterion_CE(outputs[3], targets)
        ###################################################################################
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs[3].data, 1)
        total += targets.size(0)

        correct += predicted.eq(targets.data).cpu().sum().float().item()


        b_idx = batch_idx

    print('Train s1 \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc net: %.3f%%|' % (train_loss / (b_idx + 1), 100. * correct / total))
    return train_loss / (b_idx + 1), correct / total


if __name__ == '__main__':
    time_log = datetime.now().strftime('%m-%d %H:%M')
    if int(args.log_time) :
        folder_name = 'teacher_{}'.format(time_log)


    path = os.path.join(EXPERIMENT_NAME, folder_name)
    if not os.path.exists('ckpt/' + path):
        os.makedirs('ckpt/' + path)
    if not os.path.exists('logs/' + path):
        os.makedirs('logs/' + path)

    # Save argparse arguments as logging
    with open('logs/{}/commandline_args.txt'.format(path), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # Instantiate logger
    logger = SummaryLogger(path)


    for epoch in range(RESUME_EPOCH, FINAL_EPOCH+1):
        f = open(os.path.join("logs/" + path, 'log.txt'), "a")

        ### Train ###
        train_loss, acc = train(model, epoch)
        scheduler.step()


        ### Evaluate  ###
        val_loss, test_acc  = eval(model)


        f.write('EPOCH {epoch} \t'
                'ACC_net : {acc_net:.4f} \t  \n'.format(epoch=epoch, acc_net=test_acc)
                )
        f.close()

    utils.save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, True, 'ckpt/' + path, filename='Model_{}.pth'.format(epoch))

