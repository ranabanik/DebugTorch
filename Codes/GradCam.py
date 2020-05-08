import torch
import torch.nn as nn
from torchvision.models import vgg19
from torchvision import transforms, datasets
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# model = vgg19(pretrained=False)
# print(model)

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# define a 1 image dataset
dataset = datasets.ImageFolder(root='C:\StorageDriveAll\Data\Images', transform=transform)

# define the dataloader to load that single image
dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)

class VGG(nn.Module): #inherits module properties from pytorch nn
    def __init__(self):
        super(VGG, self).__init__()
        """
        self.field = value
        The field is attached to the object itself
        """
        self.vgg = vgg19(pretrained=True)   # take the vgg19 architecture object as a field of the Network class
        self.features_conv = self.vgg.features[:36]
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.classifier = self.vgg.classifier
        self.gradients = None   # it does not have any value yet.

    def activations_hook(self, grad):   # a method to the class VGG
        self.gradients = grad

    def forward(self, x):
        x = self.features_conv(x)   # output of the 36th layer.
        h = x.register_hook(self.activations_hook)
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)

Model = VGG()
Model.eval()

img, _ = next(iter(dataloader))
pred = Model(img)#.argmax(dim=1)
print(pred.argmax(dim=1))    # takes the index of max value
# print(pred)   # pred shape: torch.Size([1, 1000]) and max value 386
if __name__ != '__main__':
    print("img:", img.shape)    # img: torch.Size([1, 3, 224, 224])
    # img = img.transpose([0, 2, 3, 1]).squeeze()
    # print(img.shape)
# Gr = Model.activations_hook()
# print(Gr)
# result_b4_classifier = Model.features_conv(img)
# print(result_b4_classifier.shape)   # torch.Size([1, 512, 14, 14])
# print(pred[0, 386])
pred[:, pred.argmax(dim=1)].backward()
gradients = Model.get_activations_gradient()
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])     # avg across 512 channels
activations = Model.get_activations(img).detach()
for i in range(512):    # weight in the channel corresponding to gradient change
    activations[:, i, :, :] *= pooled_gradients[i]
heatmap = torch.mean(activations, dim=1).squeeze()
heatmap = np.maximum(heatmap, 0)
heatmap /= torch.max(heatmap)
heatmap = heatmap.cpu().numpy()
# print(type(heatmap))    # <class 'numpy.ndarray'>
# print(heatmap)
# print('heatmap: ', heatmap.shape)   # torch.Size([14, 14])
# plt.matshow(heatmap.squeeze())
# plt.show()

imgPath = r'C:\StorageDriveAll\Data\Images\test'
imgPath = os.path.join(imgPath, 'Head1.png')
# print(imgPath)
img = cv2.imread(imgPath)
# print(img.shape)    # <class 'numpy.ndarray'> (848, 1272, 3)
# plt.imshow(img)
# plt.show()    # V
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
# print('here')
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('./Head1.jpg', superimposed_img)
# plt.imshow(superimposed_img)
# plt.show()
