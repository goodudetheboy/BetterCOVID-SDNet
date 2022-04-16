from joblib import PrintTime
from matplotlib import pyplot as plt
import torch
import numpy as np

import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from pathlib import Path

from tqdm.notebook import tqdm

from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    print(dict(Counter(dataset.targets)))

    #sampling preparation steps
    train_id, test_id = ind[spl:], ind[:spl]
    tr_sampl = SubsetRandomSampler(train_id)
    te_sampl = SubsetRandomSampler(test_id)

    #use data loader to get train and test set ready for training
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=tr_sampl,num_workers=worker)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=te_sampl,num_workers=worker)
    return (trainloader, testloader)

def run(model_name, net, optimizer, scheduler , criterion, num_epochs, trainloader, testloader):
    print(f'Training on {device}')

    best_accuracy = float('-inf')
    train_losses = []
    train_accu = []
    test_losses = []
    test_accu = []
    # Train the model
    total_step = len(trainloader)
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0
        correct = 0
        total = 0
        net.train()
        test_loss = 0
        test_accuracy = 0

        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            train_loss=running_loss/len(trainloader)
            train_accuracy = 100.*correct/total
            
            if i % 5 == 4:
                print ('Epoch [{}/{}], Step [{}/{}], Accuracy: {:.3f}, Train Loss: {:.4f}'
                .format(epoch+1, num_epochs, i+1, total_step, train_accuracy, loss.item()))
        scheduler.step()
            
        if epoch % 5 == 4:
            print('')
            print('===================Validating===================')
            net.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                running_loss = 0
                nb_classes = 2
                confusion_matrix = torch.zeros(nb_classes, nb_classes)
                for images, labels in testloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = net(images)
                    loss= criterion(outputs,labels)
                    running_loss+=loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    test_loss=running_loss/len(testloader)
                    test_accuracy = (correct*100)/total
                    for t, p in zip(labels.view(-1), predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

                class_acc = (confusion_matrix.diag()/confusion_matrix.sum(1)).tolist()
                print('Epoch: %.0f | Test Loss: %.3f | Accuracy: %.3f'%(epoch+1, test_loss, test_accuracy))
                print(f'Class accuracy: {class_acc}')
            print('')
    
        if test_accuracy > best_accuracy:
            Path('model_store/').mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), 'model_store/' + model_name + 'best-model-parameters.pt')

        for p in optimizer.param_groups:
                print(f"Epoch {epoch+1} Learning Rate: {p['lr']}")

        path = 'checkpoints/checkpoint{:04d}.pth.tar'.format(epoch)
        Path('checkpoints/').mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'epoch': num_epochs,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, path
        )

        train_accu.append(train_accuracy)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accu.append(test_accuracy)

        print("-----------------------------------------------")

    return train_accu,test_accu, test_losses, train_losses 
