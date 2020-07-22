

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
import utils
from Models import *


parser = argparse.ArgumentParser(description='Quantization finetuning for CIFAR100')
parser.add_argument('--text', default='log.txt', type=str)
parser.add_argument('--exp_name', default='cifar10/Paraphraser', type=str)
parser.add_argument('--log_time', default='1', type=str)
parser.add_argument('--lr', default='0.1', type=float)
parser.add_argument('--resume_epoch', default='0', type=int)
parser.add_argument('--epoch', default='5', type=int)
parser.add_argument('--decay_epoch', default=[150, 225], nargs="*", type=int)
parser.add_argument('--w_decay', default='5e-4', type=float)
parser.add_argument('--cu_num', default='0', type=str)
parser.add_argument('--seed', default='1', type=str)
parser.add_argument('--load_pretrained', default='trained/Teacher.pth', type=str)
parser.add_argument('--save_model', default='ckpt.t7', type=str)
parser.add_argument('--rate', type=float, default=0.5, help='The paraphrase rate k')
parser.add_argument('--beta', type=int, default=500)


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
RATE = args.rate

model = ResNet56()

# Load the teacher network
if len(args.load_pretrained) > 2 :
    path = args.load_pretrained
    state = torch.load(path)
    utils.load_checkpoint(model, state)


# According to CIFAR
Paraphraser_t = Paraphraser(64, int(round(64*RATE)))
model.to(DEVICE)
Paraphraser_t.to(DEVICE)

# Loss and Optimizer
optimizer = optim.SGD(Paraphraser_t.parameters(), lr=base_lr, momentum=0.9, weight_decay=W_DECAY)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
criterion = nn.L1Loss()


def train(model,module, epoch):
    epoch_start_time = time.time()
    print('\n EPOCH: %d' % epoch)
    model.eval()
    module.train()

    train_loss = 0
    correct = 0
    total = 0
    global optimizer

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()

        # train Paraphraser with last layer of Teacher network.
        ###################################################################################

        outputs= model(inputs)
        # reconstructed feature maps (Mode 0; see FeatureProjection.py)
        output_p= module(outputs[2],0)
        loss = criterion(output_p, outputs[2].detach())

        ###################################################################################

        loss.backward()
        optimizer.step()

        train_loss += loss.item()



        b_idx = batch_idx

    print('Train s1 \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | ' % (train_loss / (b_idx + 1)))
    return train_loss / (b_idx + 1)

if __name__ == '__main__':
    time_log = datetime.now().strftime('%m-%d %H:%M')
    if int(args.log_time) :
        folder_name = 'paraphraser_{}'.format(time_log)


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

        ### Train ###
        train_loss = train(model,Paraphraser_t, epoch)
        scheduler.step()



    utils.save_checkpoint({
        'epoch': epoch,
        'state_dict': Paraphraser_t.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, True, 'ckpt/' + path, filename='Module_{}.pth'.format(epoch))