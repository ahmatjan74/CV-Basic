import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data_utils

import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# net
test_data = datasets.MNIST(
    root='mnist',
    train=False,
    transform=transforms.ToTensor(),
    download=False
)

test_loader = data_utils.DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=True
)

cnn = torch.load('model/model.pkl')
cnn = cnn.to(device)

# loss
# eval / test
loss_test = 0
accuracy = 0

for i, (images, labels) in enumerate(test_loader):
    images = images.to(device)
    labels = labels.to(device)
    
    optput = cnn(images)
    _, pred = optput.max(1)
    
    accuracy += (pred == labels).sum().item()
    
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    pred = pred.cpu().numpy()
    
    for idx in range(images.shape[0]):
        im_data = images[idx]
        im_label = labels[idx]
        im_pred = pred[idx]
        print(im_label, im_pred)
        
        im_data = im_data.transpose(1, 2, 0)
        cv2.imshow('result: {}-{}'.format(im_label, im_pred), im_data)
        cv2.waitKey(0)
        
accuracy = accuracy / len(test_data)
print(accuracy)
        
    
    
    
    
