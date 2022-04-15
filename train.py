from matplotlib import pyplot as plt
import torch
from tqdm.notebook import tqdm

def eval(net, testloader, device):
    net.eval()
    accuracy = 0.0
    total = 0.0

    # for class accuracy
    nb_classes = 5
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
            # images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    print(confusion_matrix)
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    class_accuracy = (confusion_matrix.diag()/confusion_matrix.sum(1)).tolist()
    return accuracy, class_accuracy


def train(net, optimizer, criterion, epochs, trainloader, testloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Training on {device}')
    net.to(device)
    losses = []
    
    eval(net, testloader, device)
    return
    for epoch in tqdm(range(epochs)):
        Loss = 0.0
        count = 0
        for iter, data in enumerate(trainloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            predicted = net(images)
            _, pre1 = torch.max(predicted,dim=1)

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
        if epoch%5==0:
            overall, class_acc = eval(net, testloader, device)
            print(f'Accuracy of the network on the 10000 validation x: {overall:.4f} %')
            print(f'Class accuracy: {class_acc}')

    plt.plot(losses)

