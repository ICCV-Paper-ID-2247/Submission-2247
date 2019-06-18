"""
Created on Sun Mar 24 17:51:08 2019

@author: anonymous
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from resnet_model import *  # Imports the ResNet Model

num_classes=10

model = resnet(num_classes=num_classes,depth=110)
if True:
    model = nn.DataParallel(model).cuda()


r_model= 'Models_PCL_Only/CIFAR10_PCL_Only.pth.tar'  # the model trained with only PC Loss (without joint supervision)


checkpoint = torch.load(r_model)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                         download=True, 
                                         transform=transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                         ]))
test_loader = torch.utils.data.DataLoader(testset, batch_size=256, pin_memory=True,
                                          shuffle=False, num_workers=4)


import numpy as np
cuda = torch.cuda.is_available()
train_dataset = torchvision.datasets.CIFAR10('./data/cifar10', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                             ]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, pin_memory=True,
                                          shuffle=False, num_workers=4)

def extract_embeddings(dataloader, model): # for extracting the train embeddings and their mean
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 1024))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            _,_,__,o = model(images)
            embeddings[k:k+len(images)]= __.data.cpu().numpy()  # Penultimate Layer Embeddings
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

train_embeddings_otl, train_labels_otl = extract_embeddings(train_loader, model)

val_embeddings_otl, val_labels_otl = extract_embeddings(test_loader, model)

def extract_mean_feats(embeddings, targets):
    mean_feats= np.zeros((10,1024))
    mean_labels= np.zeros((10))
    for i in range(10): # 10 classes
        inds = np.where(targets==i)[0]
        mean_feats[i]= np.average(embeddings[inds], axis=0)
        mean_labels[i]=i
    return mean_feats, mean_labels  

train_mean_feats, train_mean_labels= extract_mean_feats(train_embeddings_otl, train_labels_otl)

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(train_mean_feats, train_mean_labels)

predicted= knn_model.predict(val_embeddings_otl) 

a=np.where(predicted==val_labels_otl)
print ('Accuracy on Clean Images is: {0:.3%} '.format( len(a[0])/len(val_labels_otl) , ' %'))







