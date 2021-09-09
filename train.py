## Imports generales
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import PIL.Image as Image
import pandas as pd
import numpy as np
import random
import sys
import os

## Imports de torch


from torchvision.transforms import Compose,PILToTensor,Normalize,ColorJitter,RandomAffine,Resize,RandomCrop,GaussianBlur
from delta_tb.deltatb.networks.net_segnetV2 import segnet
import torchvision.models as models
import torch
from CESAR_segnet import Seg_CESAR




##########################################################################
##########################################################################
##########################################################################

### Busqueda de imágenes

if len(sys.argv) > 1:
	FILES_PATH = sys.argv[1].replace("\\","/")
else:
	FILES_PATH = "./Base de datos/OpticDiscs/"
	print("Buscando base de datos en {}".format(FILES_PATH))

good = os.path.join(FILES_PATH,"Good images")
bad = os.path.join(FILES_PATH,"Neovessels")

Bad =  pd.DataFrame({"imageFilename":[os.path.join(bad,File) for File in os.listdir(bad) if ".jpeg" in File]})
Good = pd.DataFrame({"imageFilename":[os.path.join(good,File) for File in os.listdir(good) if ".jpeg" in File]}).sample(len(Bad),random_state = 42)

data = pd.concat((Good,Bad),axis = 0)
#data.head()
data['class'] = data['imageFilename'].apply(lambda x : 1 if "Good" in x else 0)
data = data.reset_index()
data = data[['imageFilename','class']]
#data.head()


Train,Validation = train_test_split(data,test_size = 0.60,random_state = 65)
X_train = Train['imageFilename']
y_train = Train['class']
#Split data
X_validation, X_test, y_validation, y_test = train_test_split(Validation['imageFilename'], Validation['class'], test_size=0.50, random_state=65)

print("Datos de entrenamiento:")
print(y_train.value_counts())
print("Datos de validación:")
print(y_validation.value_counts())
print("Datos de test:")
print(y_test.value_counts())

test_data = pd.DataFrame({'image':X_test,'class':y_test})
test_data['image'] = test_data['image'].apply(lambda x : x.split('/')[-1])
#test_data.head()

test_data.to_csv('test_data_NEW.csv',index=False)

### NETWORK PARAMETERS

INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 1
PRETRAINED = True

CUDA = True


if CUDA:
    model = segnet(INPUT_CHANNELS,OUTPUT_CHANNELS,True).cuda(0)
else:
    model = segnet(INPUT_CHANNELS,OUTPUT_CHANNELS,True)



## Load SEGNET model
model.load_from_filename("./models/model_best_512_LAST.pth")

MODEL = Seg_CESAR(model)



# Training parameters
EPOCHS = 700
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
IMAGE_SIZE = 512


# Transformaciones

# Color distribution
NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# Color modification
SATURATION = 0.2
CONTRAST = 0.2

# Affine transformations
ROTATION = (-120,120)
SCALE = (0.6,1.4)
TRANSLATE = None


INPUT_TRANSFORMATIONS = Compose([Normalize(NORMALIZATION[0],NORMALIZATION[1]),
                                 RandomAffine(degrees = ROTATION,scale = SCALE, translate = TRANSLATE),
                                 Resize((IMAGE_SIZE,IMAGE_SIZE)),
                                 GaussianBlur(3)])

TEST_TRANSFORMATIONS = Compose([Resize((IMAGE_SIZE,IMAGE_SIZE)),
                                Normalize(NORMALIZATION[0],NORMALIZATION[1]),
                                GaussianBlur(3)])



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("El modelo tiene {} parametros".format(count_parameters(MODEL)))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL.to(device);
seed = 17
torch.manual_seed(seed)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)

network_name = "DENSENET_FINAL"

random.seed(seed)


batch_size = BATCH_SIZE  # I will use batch size of 1 to keep the ratio of each image

TRAINING_acc = []
VALIDATION_acc = []
BEST_val_acc = 0.0

for epoch in range(EPOCHS):
    indexes = [idx for idx in range(len(X_train))]
    pbar = tqdm(range(len(X_train)//batch_size),ncols = 100)
    running_loss = 0.0
    running_acc = 0.0
    t = 0
    
    for step in pbar:
        # Load data
        idx = random.sample(indexes,batch_size)
        X = X_train.iloc[idx]
        y = y_train.iloc[idx]

        # Remove indexes
        [indexes.remove(i) for i in idx]

        # Load images
        try:
            images = [torch.transpose(torch.FloatTensor(np.array(Image.open(File))),2,0) for File in X]
        except:
            print("Images not loaded")
            continue
        # Load y_true
        y_true = torch.LongTensor([c for c in y]).to(device)
        
        # Convert images to tensor
        x_batch = torch.FloatTensor().to(device)
        for image in images:
            P = INPUT_TRANSFORMATIONS(image)
            P = P.unsqueeze(0).to(device)
            x_batch = torch.cat((x_batch,P))
        
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = MODEL(x_batch)
        loss = criterion(outputs, y_true)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        t += batch_size

        _, preds = torch.max(outputs, 1)
        running_acc += torch.sum(preds == y_true).cpu().detach().numpy()
        acc = torch.sum(preds == y_true).cpu().detach().numpy()/batch_size;
        pbar.set_description("Epoch: {} Accuracy: {:0.5f} Loss: {:0.5f} ".format(epoch+1,running_acc/t,loss.item()))
    #Validation
    TRAINING_acc.append(running_acc/t)
    val_acc = 0.0
    val_loss = 0.0
    t = 0
    for point in range(len(X_validation)//batch_size):
        with torch.no_grad():

            X = X_validation.iloc[point*batch_size:(point+1)*batch_size]
            y = y_validation.iloc[point*batch_size:(point+1)*batch_size]


            # Load images
            try:
                images = [torch.transpose(torch.FloatTensor(np.array(Image.open(File))),2,0) for File in X]
            except:
                print("Error loading images")
                continue
            # Load y_true
            y_true = torch.LongTensor([c for c in y]).to(device)
            
            # Convert images to tensor
            x_batch = torch.FloatTensor().to(device)
            for image in images:
                P = INPUT_TRANSFORMATIONS(image).unsqueeze(0).to(device)
                x_batch = torch.cat((x_batch,P))

            
            outputs = MODEL(x_batch)
            loss = criterion(outputs, y_true)
            val_loss += loss.item()
            t += batch_size
            _, preds = torch.max(outputs, 1)
            val_acc += torch.sum(preds == y_true).cpu().detach().numpy()
    VALIDATION_acc.append(val_acc/t)
    print("Validation -- Accuracy: {:0.5f} Loss: {:0.5f} \n".format(val_acc/t,loss.item()))
    if val_acc/t > BEST_val_acc:
        try:
            torch.save(MODEL,"./content/checkpoint_{}_{}_{:0.5f}.ph".format(network_name,epoch+1,val_acc/t))
            BEST_val_acc = val_acc/t
        except:
            continue

history = pd.DataFrame({"Train":TRAINING_acc,"Validation":VALIDATION_acc})
history.to_csv('{}_history.csv'.format(network_name))