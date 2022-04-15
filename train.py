from matplotlib import pyplot as plt
import torch
import numpy as np

import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from tqdm.notebook import tqdm


def preprocess_data(directory:str, batch_size:int, test_size:int, rand_num:int, worker:int):
    '''
        directory: the directory of processed directory with class folders inside
        batch_size: size of batch for training
        test_size: percent of dataset used for test
        rand_num: put random number for reproducibility
        worker: number of worker in computation
        
        return train and test data ready for training
    '''
    #pipeline to resize images, crop, convert to tensor, and normalize
    trans = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])
    
    dataset = torchvision.datasets.ImageFolder(root=directory, transform=trans) #read image in folder to data with labels
    
    train_len = len(dataset) #get length of whole data
    ind = list(range(train_len)) #indices of whole data
    spl = int(np.floor(test_size * train_len)) #index of test data
    
    #reproducibility and shuffle step
    np.random.seed(rand_num) 
    np.random.shuffle(ind)
    
    #sampling preparation steps
    train_id, test_id = ind[spl:], ind[:spl]
    tr_sampl = SubsetRandomSampler(train_id)
    te_sampl = SubsetRandomSampler(test_id)

    #use data loader to get train and test set ready for training
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=tr_sampl,num_workers=worker)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=te_sampl,num_workers=worker)
    return (trainloader, testloader)

def eval(net, testloader, device):
    net.eval()
    accuracy = 0.0
    total = 0.0

    # for class accuracy
    nb_classes = 5
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    
    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    class_accuracy = (confusion_matrix.diag()/confusion_matrix.sum(1)).tolist()
    return accuracy, class_accuracy


def train(net, optimizer, criterion, epochs, trainloader, testloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Training on {device}')
    net.to(device)
    losses = []
    
    for epoch in tqdm(range(epochs)):
        Loss = 0.0
        count = 0
        for iter, data in enumerate(trainloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            predicted = net(images)
            optimizer.zero_grad()
            loss = criterion(predicted, labels)
            loss.backward()
            optimizer.step()

            Loss += loss.item() #accumulate the loss
            count += 1

        avg_loss = Loss/count
        losses.append(avg_loss) #append the average loss for each batch
        print('Epoch:[{}/{}], training loss: {:.4f}'.format(epoch+1, epochs, avg_loss))

        # validation
        if epoch%5==4:
            overall, class_acc = eval(net, testloader, device)
            print(f'Accuracy of the network on validation set: {overall:.4f} %')
            print(f'Class accuracy: {class_acc}')

    plt.plot(losses)

