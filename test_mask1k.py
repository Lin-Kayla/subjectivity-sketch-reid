from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import TestData, TestData_multi, Mask1kData_multi, Mask1kData_single, TestData_ensemble
from data_manager import *
from eval_metrics import eval
from model import model
from utils import *
from loss import OriTripletLoss, TripletLoss_WRT
from tensorboardX import SummaryWriter
import clip

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
# training
parser.add_argument('--gpu', default='0', type=str, 
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--save_epoch', default=1, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--train_mq', action='store_true', help='train multi-query')
parser.add_argument('--test_mq', action='store_true', help='test multi-query')
# dataset
parser.add_argument('--dataset', default='mask1k', 
                    help=' dataset name: mask1k (short for Market-Sketch-1K) or pku (short for PKU-Sketch)')
parser.add_argument('--data_path', default='/data3/lkj/rebuttal_sketchreid/dataset/market-mix-cross/', type=str, 
                    help='path to dataset, and where you store processed attributes')
parser.add_argument('--train_style', default='A', type=str, 
                    help='using which styles as the trainset, can be any combination of A-F')
parser.add_argument('--test_style', default='AB', type=str, 
                    help='using which styles as the testset, can be any combination of A-F')
# optimizer
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
# model
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
# hyper-parameters
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
data_path = args.data_path
log_path = args.log_path + 'mask1k/'

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = f'{args.dataset}_{args.train_style}_{args.test_style}'
sys.stdout = Logger(log_path + suffix + '_os.txt')
vis_log_dir = args.vis_log_path + suffix + '/'
os.makedirs(vis_log_dir, exist_ok=True)
writer = SummaryWriter(vis_log_dir)

print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
# training set
if args.train_mq:
    trainset = Mask1kData_multi(data_path, args.train_style,  transform=transform_train)
else:
    trainset = Mask1kData_single(data_path, args.train_style, transform=transform_train)

# generate the idx of each person identity
color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_sketch_label)

# testing set
if len(args.test_style) == 1:
    # test single-query & single style
    query_img, query_label = process_test_mask1k_single(data_path, test_style=args.test_style)
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
elif args.test_mq:
    # test multi-query & multi styles
    query_img, query_label = process_test_mask1k_multi(data_path, args.test_style)
    queryset = TestData_multi(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
else:
    # test single-query & multi styles
    query_img, query_label, query_style = process_test_market_ensemble(data_path, test_style=args.test_style)
    queryset = TestData_ensemble(query_img, query_label, query_style, transform=transform_test, img_size=(args.img_w, args.img_h))

gall_img, gall_label = process_test_market(data_path, modal='photo')
gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  sketch  | {:5d} | {:8d}'.format(n_class, len(trainset.train_sketch_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
model_clip, preprocess = clip.load("ViT-B/32", device=device)

net = model(n_class, model_clip, args.batch_size, args.num_pos, arch=args.arch, train_multi_query=args.train_mq, test_multi_query = args.test_mq)
net.to(device)
cudnn.benchmark = True

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion_id = nn.CrossEntropyLoss()
criterion_tri = TripletLoss_WRT()

criterion_id.to(device)
criterion_tri.to(device)


if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters())) \
                        +list(map(id, net.clip.parameters())) \
                            + list(map(id, net.cmalign.maskFc.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer1 = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr},
        {'params': net.cmalign.maskFc.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)
    optimizer2 = optim.Adam(
        [{'params': net.clip.parameters(), 'lr': 5e-5}],
        betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr

def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            if torch.cuda.is_available():    
                input = Variable(input.cuda())
            else:
                input = Variable(input)
            feat = net(input, input, None, None, 1)['feat4_p_norm']
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    styles = np.zeros((nquery))
    if len(args.test_style) == 1:
        with torch.no_grad():
            for batch_idx, (input, label) in enumerate(query_loader):
                batch_num = input.size(0)
                input = Variable(input.to(device))
                feat = net(input, input, None, None, 2)['feat4_p_norm']
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                ptr = ptr + batch_num
    elif args.test_mq:
        with torch.no_grad():
            for batch_idx, (input, label, style) in enumerate(query_loader):
                batch_num = input.size(0)
                if torch.cuda.is_available():
                    input = Variable(input.cuda())
                else:
                    input = Variable(input)
                feat = net(input, input, None, style, 2)['feat4_p_norm']
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                ptr = ptr + batch_num
    else:
        with torch.no_grad():
            for batch_idx, (input, label, style) in enumerate(query_loader):
                batch_num = input.size(0)
                if torch.cuda.is_available():
                    input = Variable(input.cuda())
                else:
                    input = Variable(input)
                feat = net(input, input, None, None, 2)['feat4_p_norm']
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                styles[ptr:ptr + batch_num] = np.array([ord(i)-ord('A') for i in list(style)])
                ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))

    # evaluation
    cmc, mAP, mINP      = eval(-distmat, query_label, gall_label)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    return cmc, mAP, mINP

if __name__ == '__main__':
    cmc, mAP, mINP = test(start_epoch)

    print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    print('Best Epoch [{}]'.format(start_epoch))

