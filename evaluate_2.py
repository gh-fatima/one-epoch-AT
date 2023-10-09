############
## Import ##
############
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from model.model import encoder
from dataset.datasets import load_dataset
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch
import numpy as np
from func import WeightedKNNClassifier, linear_train
import torchvision.transforms as transforms
from dataset.aug import ContrastiveLearningViewGenerator
import torchvision
######################
## Parsing Argument ##
######################
import argparse
parser = argparse.ArgumentParser(description='Evaluation')

parser.add_argument('--test_patches', type=int, default=1,
                    help='number of patches used in testing (default: 128)')  

parser.add_argument('--data', type=str, default="cifar10",
                    help='dataset (default: cifar10)')  

parser.add_argument('--arch', type=str, default="resnet18-cifar",
                    help='network architecture (default: resnet18-cifar)')

parser.add_argument('--lr', type=float, default=0.03,
                    help='learning rate for linear eval (default: 0.03)')       

parser.add_argument('--linear', type=bool, default=True,
                    help='use linear eval or not')

parser.add_argument('--knn', help='evaluate using kNN measuring cosine similarity', action='store_true')

parser.add_argument('--model_path', type=str, default="",
                    help='model directory for eval')

parser.add_argument('--scale_min', type=float, default=0.08, 
                    help='Minimum scale for resizing')

parser.add_argument('--scale_max', type=float, default=1.0, 
                    help='Maximum scale for resizing')

parser.add_argument('--ratio_min', type=float, default=0.75, 
                    help='Minimum aspect ratio')

parser.add_argument('--ratio_max', type=float, default=1.3333333333333333, 
                    help='Maximum aspect ratio')

parser.add_argument('--type', type=str, default="crop",
                    help='crop vs. patch')

parser.add_argument('--epochs', type=int, default=200,
                    help='max number of epochs to finish')  

parser.add_argument('--bs_centralcrop_train', type=int, default =256,
                    help='batchSize for training central_crop') 

parser.add_argument('--bs_centralcrop_test', type=int, default =100,
                    help='batchSize for test central_crop') 


parser.add_argument('--bs_patch_train', type=int, default = 50,
                    help=' batchSize for testing ') 

parser.add_argument('--bs_patch_test', type=int, default = 16,
                    help=' batchSize for testing ') 

parser.add_argument('--alpha', type=float, default = 1e-2, 
                    help='movement multiplier per iteration in adversarial examples')

parser.add_argument('--iter', type=int, default = 20, 
                    help='number of iterations for generating adversarial Examples')

parser.add_argument('--hidden_units', type=int, default = 4096, 
                    help='number of iterations for generating adversarial Examples')



            
args = parser.parse_args()

print("Running with test_patches = " + str(args.test_patches) + "model_path = " + args.model_path + "/type = " + args.type)

######################
## Testing Accuracy ##
######################
test_patches = args.test_patches

if args.data=='cifar10':
     args.num_class = 10
else:
     args.num_class = 100

def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size

# knn_classifier = WeightedKNNClassifier()


def chunk_avg(x,n_chunks=2,normalize=False):
    x_list = x.chunk(n_chunks,dim=0)
    x = torch.stack(x_list,dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0),dim=1)


def train_Eval(net, train_loader):
    
    train_z_full_list, train_y_list, test_z_full_list, test_y_list = [], [], [], []
    
    with torch.no_grad():
        for x, y in tqdm(train_loader):

            x = torch.cat(x, dim = 0)
            
            z_proj, z_pre = net(x, is_test=True)
            z_pre = chunk_avg(z_pre, test_patches)
            z_pre = z_pre.detach().cpu()
            train_z_full_list.append(z_pre)
            
            
            # knn_classifier.update(train_features = z_pre, train_targets = y)

            train_y_list.append(y)
                
    train_features_full, train_labels = torch.cat(train_z_full_list,dim=0), torch.cat(train_y_list,dim=0)
   
        
 
    LL = linear_train(train_features_full, train_labels, lr=args.lr, num_classes = args.num_class)
    return LL
    

    
def chunk_avg(x,n_chunks=2,normalize=False):
    x_list = x.chunk(n_chunks,dim=0)
    x = torch.stack(x_list,dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0),dim=1)



#Get Dataset
if args.data == "imagenet100" or args.data == "imagenet":
        
    memory_dataset = load_dataset(args, args.data, train=True, num_patch = test_patches)
    memory_loader = DataLoader(memory_dataset, batch_size=args.bs_patch_train, shuffle=True, drop_last=True,num_workers=8)

    test_data = load_dataset(args, args.data, train=False, num_patch = test_patches)
    test_loader = DataLoader(test_data, batch_size=args.bs_patch_train, shuffle=True, num_workers=8)

else:
    print(args.data)
    memory_dataset = load_dataset(args, args.data, train=True, num_patch = test_patches)
    memory_loader = DataLoader(memory_dataset, batch_size=args.bs_patch_train, shuffle=True, drop_last=True,num_workers=8)

    # test_data = load_dataset(args.data, train=False, num_patch = test_patches)
    # test_loader = DataLoader(test_data, batch_size=50, shuffle=True, num_workers=8)

# Load Model and Checkpoint
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
net = encoder(arch = args.arch)
net = nn.DataParallel(net)

save_dict = torch.load(args.model_path)
net.load_state_dict(save_dict,strict=False)
net.cuda()
net.eval()
LL = train_Eval(net, memory_loader)
LL.cuda()


# class EncoderWithHead(nn.Module):
#     def __init__(self, encoder, head):
#         super(EncoderWithHead, self).__init__()
#         self.encoder        = encoder
#         self.head = head    
#     def forward(self, x):
#         _,fea = self.encoder(x)
#         out = self.head(fea)
#         return out


print('###################Train Lucun, Test Lecun Model#############')
testTransform = ContrastiveLearningViewGenerator(num_patch=args.test_patches, scale_min = args.scale_min, scale_max = args.scale_max, ratio_min = args.ratio_min, ratio_max = args.ratio_max)
if args.data == 'cifar10':
    testDataset        = torchvision.datasets.CIFAR10(root='./data/' ,train=False, transform=testTransform, download=True)
elif args.data == 'cifar100':
    testDataset        = torchvision.datasets.CIFAR100(root='./data/' ,train=False, transform=testTransform, download=True) 

batchSize = args.bs_patch_test
testLoader      = torch.utils.data.DataLoader(dataset=testDataset,  batch_size=batchSize, num_workers=4, pin_memory=True, shuffle=True, drop_last=True )

def testEvalNet_Ro():
    net.eval()
    LL.eval()
    total_acc_test = 0
    for i, (X, labels) in tqdm(enumerate(testLoader)):
            
            X = torch.cat(X, dim = 0).to(device)
            labels = labels.to(device)
            z_proj, z_pre = net(X, is_test=True)
            z_pre = chunk_avg(z_pre, test_patches)
            yp = LL(z_pre).to(device)
            total_acc_test += (yp.max(dim=1)[1] == labels).sum().item()
    print('Acc_Test =', total_acc_test / len(testLoader.dataset),sep="\t")
    return total_acc_test / len(testLoader.dataset)

testEvalNet_Ro()



def pgd_linf_end_lecun(BE,my_LL, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        _,z_pre =  BE(X+delta, is_test=True)
        z_pre = chunk_avg(z_pre, test_patches)
        z_pre = z_pre.to(device)
        yp = my_LL(z_pre).to(device)
        loss = nn.CrossEntropyLoss()(yp, y)
        loss.backward()
        with torch.no_grad():
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.data = torch.clamp(X + delta.data, min=0, max=1) - X
        delta.grad.zero_()
    return delta.detach()


def pgd_linf_end_lecun_iid(BE,my_LL, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    y = torch.repeat_interleave(y,test_patches)
    for t in range(num_iter):
        _,z_pre =  BE(X+delta, is_test=True)
        # z_pre = chunk_avg(z_pre, test_patches)
        z_pre = z_pre.to(device)
        yp = my_LL(z_pre).to(device)
        loss = nn.CrossEntropyLoss()(yp, y)
        loss.backward()
        with torch.no_grad():
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.data = torch.clamp(X + delta.data, min=0, max=1) - X
        delta.grad.zero_()
    return delta.detach()

print('################### Train Lecun, Adv Test Lecun#############')
# testTransform = ContrastiveLearningViewGenerator(num_patch=128)
# testDataset        = torchvision.datasets.CIFAR10(root='./data/' ,train=False, transform=testTransform, download=True)

# batchSize = 100

# testLoader      = torch.utils.data.DataLoader(dataset=testDataset,  batch_size=batchSize, num_workers=4, pin_memory=True, shuffle=True, drop_last=True )

def testEvalNet_adv(eps):
    net.eval()
    LL.eval()
    total_acc_test = 0
    for i, (X, labels) in tqdm(enumerate(testLoader)):
            X = torch.cat(X, dim = 0).to(device)
            labels = labels.to(device)
            delta = pgd_linf_end_lecun(net,LL, X, labels, eps, args.alpha, args.iter)
            _,z_pre =  net(X+delta, is_test=True)
            z_pre = chunk_avg(z_pre, test_patches)
            z_pre = z_pre.to(device)
            yp = LL(z_pre).to(device)
            total_acc_test += (yp.max(dim=1)[1] == labels).sum().item()
    print('Acc_Test =', total_acc_test / len(testLoader.dataset),sep="\t")
    return total_acc_test / len(testLoader.dataset)

testEvalNet_adv(4/255)
testEvalNet_adv(8/255)
testEvalNet_adv(16/255)
# print('Result on clean data =',r)
print('###################train Lecun, test adv_iid#############')
def testEvalNet_adv(eps):
    net.eval()
    LL.eval()
    total_acc_test = 0
    for i, (X, labels) in tqdm(enumerate(testLoader)):
            X = torch.cat(X, dim = 0).to(device)
            labels = labels.to(device)
            delta = pgd_linf_end_lecun_iid(net,LL, X, labels, eps, args.alpha, args.iter)
            _,z_pre =  net(X+delta, is_test=True)
            z_pre = chunk_avg(z_pre, test_patches)
            z_pre = z_pre.to(device)
            yp = LL(z_pre).to(device)
            total_acc_test += (yp.max(dim=1)[1] == labels).sum().item()
    print('Acc_Test =', total_acc_test / len(testLoader.dataset),sep="\t")
    return total_acc_test / len(testLoader.dataset)

# testEvalNet_adv(4/255)
# testEvalNet_adv(8/255)
# testEvalNet_adv(16/255)
# print('Result on clean data =',r)
# print('###################Train Lecun, Test Normal#############')
# testTransform = transforms.Compose([
#         transforms.ToTensor()])
# testDataset        = torchvision.datasets.CIFAR10(root='./data/' ,train=False, transform=testTransform, download=True)

# batchSize = args.bs_centralcrop_train

# testLoader      = torch.utils.data.DataLoader(dataset=testDataset,  batch_size=batchSize, num_workers=4, pin_memory=True, shuffle=True, drop_last=True )

# def testEvalNet1():
#     net.eval()
#     LL.eval()
#     total_acc_test = 0
#     for i, (X, labels) in tqdm(enumerate(testLoader)):
#             X = X.to(device)
#             labels = labels.to(device)
#             _,z_pre =  net(X, is_test=True)
#             z_pre = z_pre.to(device)
#             yp = LL(z_pre).to(device)
#             total_acc_test += (yp.max(dim=1)[1] == labels).sum().item()
#     print('Acc_Test =', total_acc_test / len(testLoader.dataset),sep="\t")
#     return total_acc_test / len(testLoader.dataset)
# r = testEvalNet1()
# print('Result on clean data =',r)



print('###################Training EvalNet#############')
# 
# normalize = transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
trainEvalTransform = transforms.Compose([transforms.ToTensor()])
if args.data == 'cifar10':
    trainEvalDataset        = torchvision.datasets.CIFAR10(root='./data/' ,train=True, transform=trainEvalTransform, download=True)
elif args.data=='cifar100':
    trainEvalDataset        = torchvision.datasets.CIFAR100(root='./data/' ,train=True, transform=trainEvalTransform, download=True)

batchSize = args.bs_centralcrop_train
trainEvalLoader      = torch.utils.data.DataLoader(dataset=trainEvalDataset ,  batch_size=batchSize, num_workers=4, pin_memory=True, shuffle=True, drop_last=True )

num_class = args.num_class
LL2 = nn.Linear(args.hidden_units,num_class).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(LL2.parameters(), lr=3e-4)
numEpochs = args.epochs
def trainEvalNet_Ro():
    totalStep = len(trainEvalLoader)
    net.eval()
    LL2.train()
    for epoch in range(numEpochs):
        for i, (X, labels) in enumerate(trainEvalLoader):
            X = X.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                _,z_pre =  net(X, is_test=True)
                z_pre = z_pre.to(device)
            yp = LL2(z_pre)
            loss = criterion(yp, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if (i+1) % 1 == 0:
            #     print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch+1, numEpochs, i+1, totalStep, loss.item()),flush=True)
    # PATH = './CIFAR100_EvalNet_Epoch='+str(numEpochs)+'_BatchSize='+str(batchSize)+'.pt'
    # torch.save(LL2.state_dict(), PATH)
    
trainEvalNet_Ro()   

print('###################Train normal, test normal#############')

testTransform = transforms.Compose([
        transforms.ToTensor()])

if args.data == 'cifar10':
    testDataset        = torchvision.datasets.CIFAR10(root='./data/' ,train=False, transform=testTransform, download=True)
elif args.data == 'cifar100':
    testDataset        = torchvision.datasets.CIFAR100(root='./data/' ,train=False, transform=testTransform, download=True)

batchSize = args.bs_centralcrop_test

testLoader      = torch.utils.data.DataLoader(dataset=testDataset,  batch_size=batchSize, num_workers=4, pin_memory=True, shuffle=True, drop_last=True )


def testEvalNet2():
    net.eval()
    LL2.eval()
    total_acc_test = 0
    for i, (X, labels) in tqdm(enumerate(testLoader)):
            X = X.to(device)
            labels = labels.to(device)
            _,z_pre =  net(X, is_test=True)
            z_pre = z_pre.to(device)
            yp = LL2(z_pre).to(device)
            total_acc_test += (yp.max(dim=1)[1] == labels).sum().item()
    print('Acc_Test =', total_acc_test / len(testLoader.dataset),sep="\t")
    return total_acc_test / len(testLoader.dataset)

testEvalNet2()
# print('Result on clean data =',r)

print('###################train normal, test adv normal#############')

def pgd_linf_end(BE,my_LL, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        _,z_pre =  BE(X+delta, is_test=True)
        z_pre = z_pre.to(device)
        yp = my_LL(z_pre).to(device)
        loss = nn.CrossEntropyLoss()(yp, y)
        loss.backward()
        with torch.no_grad():
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.data = torch.clamp(X + delta.data, min=0, max=1) - X
        delta.grad.zero_()
    return delta.detach()

def testEvalNet_adv_Ro_end(epst):
    totalStep = len(testLoader)
    net.eval()
    LL2.eval()
    total_acc_test = 0
    for i, (X, labels) in tqdm(enumerate(testLoader)):
            X = X.to(device)
            labels = labels.to(device)
            delta = pgd_linf_end(net, LL2, X, labels, epst, args.alpha, args.iter)
            X_adv = (X + delta)
            _,z_pre =  net(X_adv, is_test=True)
            z_pre = z_pre.to(device)
            yp = LL2(z_pre).to(device)
            total_acc_test += (yp.max(dim=1)[1] == labels).sum().item()
    print('Acc_Test =', total_acc_test / len(testLoader.dataset),sep="\t")        
    return total_acc_test/len(testLoader.dataset)

# print('Result with end-to-end attacks:')
# testEvalNet_adv_Ro_end(0)
# # testEvalNet_adv_Ro_end(2/255)
testEvalNet_adv_Ro_end(4/255)
testEvalNet_adv_Ro_end(8/255)
testEvalNet_adv_Ro_end(16/255)


print('###################train Lecun, test adv normal#############')



def testEvalNet_adv_Ro_end(epst):
    totalStep = len(testLoader)
    net.eval()
    LL.eval()
    total_acc_test = 0
    for i, (X, labels) in tqdm(enumerate(testLoader)):
            X = X.to(device)
            labels = labels.to(device)
            delta = pgd_linf_end(net, LL, X, labels, epst, args.alpha, args.iter)
            X_adv = (X + delta)
            _,z_pre =  net(X_adv, is_test=True)
            z_pre = z_pre.to(device)
            yp = LL(z_pre).to(device)
            total_acc_test += (yp.max(dim=1)[1] == labels).sum().item()
    print('Acc_Test =', total_acc_test / len(testLoader.dataset),sep="\t")        
    return total_acc_test/len(testLoader.dataset)

# print('Result with end-to-end attacks:')
# testEvalNet_adv_Ro_end(0)
# # testEvalNet_adv_Ro_end(2/255)
# testEvalNet_adv_Ro_end(4/255)
# testEvalNet_adv_Ro_end(8/255)
# testEvalNet_adv_Ro_end(16/255)

