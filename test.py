import sys
import os
import numpy as np

from torchvision.transforms import Compose,PILToTensor,Normalize,ColorJitter,RandomAffine,Resize,RandomCrop,GaussianBlur
from delta_tb.deltatb.networks.net_segnetV2 import segnet
import torchvision.models as models
import torch
from CESAR_segnet import Seg_CESAR
import PIL.Image as Image



## Seleccionar "GPU" si está disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## load image example

if len(sys.argv) > 1:
	IMAGE = sys.argv[1].replace("\\","/")
else:
	IMAGE = "./test_image.jpeg"

## Input normalization

# Color distribution
NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
IMAGE_SIZE = 512


TEST_TRANSFORMATIONS = Compose([Resize((IMAGE_SIZE,IMAGE_SIZE)),
                                Normalize(NORMALIZATION[0],NORMALIZATION[1]),
                                GaussianBlur(3)])


## Cargar modelo preentrenado
MODEL = torch.load('./content/checkpoint_DENSENET_FINAL_229_0.87500.ph')
MODEL.to(device);

# Load image
I = Image.open(IMAGE)

# Transform array to Tensor
X = torch.transpose(torch.FloatTensor(np.array(I)),2,0)
X = TEST_TRANSFORMATIONS(X).unsqueeze(0).to(device)

output = MODEL(X)

_,preds = torch.max(output, 1)
print("------------------------------------------------------")
print("Esta imagen está clasificada como {}".format("Mala" if preds.detach().cpu() == 0 else "Buena"))
print("------------------------------------------------------")