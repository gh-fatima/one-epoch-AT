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
from func import WeightedKNNClassifier , linear_train
import torchvision
from dataset.aug import ContrastiveLearningViewGenerator

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


parser.add_argument('--knn', help='evaluate using kNN measuring cosine similarity', action='store_true')

parser.add_argument('--model_path', type=str, default="",
                    help='model directory for eval')

parser.add_argument('--scale_min', type=float, default=0.25, 
                    help='Minimum scale for resizing')

parser.add_argument('--scale_max', type=float, default=0.25, 
                    help='Maximum scale for resizing')

parser.add_argument('--ratio_min', type=float, default=1, 
                    help='Minimum aspect ratio')

parser.add_argument('--ratio_max', type=float, default=1, 
                    help='Maximum aspect ratio')

parser.add_argument('--type', type=str, default="patch",
                    help='crop vs. patch')

parser.add_argument('--bs_patch_train', type=int, default = 50,
                    help=' batchSize for testing ') 

parser.add_argument('--bs_patch_test', type=int, default = 16,
                    help=' batchSize for testing ') 

parser.add_argument('--alpha', type=float, default = 1e-2, 
                    help='movement multiplier per iteration in adversarial examples')



parser.add_argument('--hidden_units', type=int, default = 4096, 
                    help='number of iterations for generating adversarial Examples')

parser.add_argument('--num_class', type=int, default = 10, 
                    help='number of classes')

parser.add_argument('--num_epochs', type=int, default = 200, 
                    help='number of epochs')
            
args = parser.parse_args()

print("Running with test_patches = " + str(args.test_patches) + "model_path = " + args.model_path + "/type = " + args.type)

######################
## Testing Accuracy ##
######################
test_patches = args.test_patches

def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size


knn_classifier = WeightedKNNClassifier()

def chunk_avg(x,n_chunks=2,normalize=False):
    x_list = x.chunk(n_chunks,dim=0)
    x = torch.stack(x_list,dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0),dim=1)


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




def train(net, train_loader):
    
    train_z_full_list, train_y_list = [], []
    
    with torch.no_grad():
        for x, y in tqdm(train_loader):

            x = torch.cat(x, dim = 0)
            _, z_pre = net(x, is_test=True)
            z_pre = chunk_avg(z_pre, test_patches)
            z_pre = z_pre.detach().cpu()
            train_z_full_list.append(z_pre)
            knn_classifier.update(train_features = z_pre, train_targets = y)
            train_y_list.append(y)    

    train_features_full, train_labels = torch.cat(train_z_full_list,dim=0), torch.cat(train_y_list,dim=0)
    LL = linear_train(train_features_full, train_labels, lr=args.lr, num_classes = args.num_class, num_epochs = args.num_epochs)

    return LL
    
def test(net, test_loader, LL, eps, iter):
    net.eval()
    LL.eval()            
    for x, y in tqdm(test_loader):
        x = torch.cat(x, dim = 0).to(device)
        y = y.to(device)
        delta = pgd_linf_end_lecun(net,LL, x, y, eps, args.alpha, iter)  
        _, z_pre = net(x+delta, is_test=True)
        z_pre = chunk_avg(z_pre, test_patches)
        z_pre = z_pre.detach().cpu()
        y = y.detach().cpu()
        knn_classifier.update(test_features = z_pre, test_targets = y)
        
     
    
    print("Using KNN to evaluate accuracy")
    top1, top5 = knn_classifier.compute()
    print("KNN (top1/top5):", top1, top5)
   


#Get Dataset
if args.data == "imagenet100" or args.data == "imagenet":
        
    memory_dataset = load_dataset(args, args.data, train=True, num_patch = test_patches)
    memory_loader = DataLoader(memory_dataset, batch_size=args.bs_patch_train, shuffle=True, drop_last=True,num_workers=8)

    test_data = load_dataset(args, args.data, train=False, num_patch = test_patches)
    test_loader = DataLoader(test_data, batch_size=args.bs_patch_test, shuffle=True, num_workers=8)

else:
    print(args.data)
    memory_dataset = load_dataset(args, args.data, train=True, num_patch = test_patches)
    memory_loader = DataLoader(memory_dataset, batch_size=args.bs_patch_train, shuffle=True, drop_last=True,num_workers=8)

    test_data = load_dataset(args,args.data,train=False, num_patch = test_patches)
    test_loader = DataLoader(test_data, batch_size=args.bs_patch_test, shuffle=True, num_workers=8)

# Load Model and Checkpoint
# Load Model and Checkpoint
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
net = encoder(arch = args.arch)
net = nn.DataParallel(net)
save_dict = torch.load(args.model_path)
net.load_state_dict(save_dict,strict=False)
net.cuda()
net.eval()

LL = train(net, memory_loader)
LL.cuda()
test(net, test_loader, LL,  0.0,0)
test(net, test_loader, LL, 4/255,20)
test(net, test_loader, LL, 8/255,20)
test(net, test_loader, LL, 16/255,20)

print('###################Train Lucun, Test Lecun Model#############')
testTransform = ContrastiveLearningViewGenerator(num_patch=args.test_patches, scale_min = args.scale_min, scale_max = args.scale_max, ratio_min = args.ratio_min, ratio_max = args.ratio_max)
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