import os
import sys
import numpy as np
import PIL
from PIL import Image
from sklearn.metrics import confusion_matrix

import torch
import torch.backends.cudnn as cudnn

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

whereIam = os.uname()[1]

print("load model")
if whereIam == "wdtim719z":
    sys.path.append("/home/optimom/github/EfficientNet-PyTorch")
    sys.path.append("/home/optimom/github/pytorch-image-models")
    sys.path.append("/home/optimom/github/pretrained-models.pytorch")
    sys.path.append("/home/optimom/github/segmentation_models.pytorch")
if whereIam in ["calculon", "astroboy", "flexo", "bender"]:
    sys.path.append("/d/achanhon/github/EfficientNet-PyTorch")
    sys.path.append("/d/achanhon/github/pytorch-image-models")
    sys.path.append("/d/achanhon/github/pretrained-models.pytorch")
    sys.path.append("/d/achanhon/github/segmentation_models.pytorch")
import segmentation_models_pytorch

with torch.no_grad():
    net = torch.load("build/model.pth")
    net = net.to(device)
    net.eval()


print("massif benchmark")
import dataloader

miniworld = dataloader.MiniWorld("test")


def accu(cm):
    return 100.0 * (cm[0][0] + cm[1][1]) / np.sum(cm)


def f1(cm):
    return 50.0 * cm[0][0] / (cm[0][0] + cm[1][0] + cm[0][1]) + 50.0 * cm[1][1] / (
        cm[1][1] + cm[1][0] + cm[0][1]
    )


cm = {}
with torch.no_grad():
    for town in miniworld.towns:
        print(town)
        cm[town] = np.zeros((2, 2), dtype=int)
        for i in range(miniworld.data[town].nbImages):
            imageraw, label = miniworld.data[town].getImageAndLabel(i)

            image = torch.Tensor(np.transpose(imageraw, axes=(2, 0, 1))).unsqueeze(0)
            globalresize = torch.nn.AdaptiveAvgPool2d((image.shape[2], image.shape[3]))
            power2resize = torch.nn.AdaptiveAvgPool2d(
                ((image.shape[2] // 32) * 32, (image.shape[3] // 32) * 32)
            )
            image = power2resize(image)

            if image.shape[2] < 512 and image.shape[3] < 512:
                pred = net(image.to(device))
            else:
                pred = dataloader.largeforward(net, image, device)

            pred = globalresize(pred)
            _, pred = torch.max(pred[0], 0)
            pred = pred.cpu().numpy()

            assert label.shape == pred.shape

            cm[town] += confusion_matrix(label.flatten(), pred.flatten(), labels=[0, 1])

            if town in ["toulouse", "potsdam"]:
                imageraw = PIL.Image.fromarray(np.uint8(imageraw))
                imageraw.save("build/" + town + "_" + str(i) + "_x.png")
                label = PIL.Image.fromarray(np.uint8(label) * 255)
                label.save("build/" + town + "_" + str(i) + "_y.png")
                pred = PIL.Image.fromarray(np.uint8(pred) * 255)
                pred.save("build/" + town + "_" + str(i) + "_z.png")

        print(cm[town][0][0], cm[town][0][1], cm[town][1][0], cm[town][1][1])
        print(
            accu(cm[town]),
            f1(cm[town]),
        )

print("-------- results ----------")
for town in miniworld.towns:
    print(town, f1(cm[town]))

globalcm = np.zeros((2, 2), dtype=int)
for town in miniworld.towns:
    globalcm += cm[town]
print("miniworld", f1(globalcm))
