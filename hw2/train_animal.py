import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import torch.utils.data as data
from torchvision import datasets,models
from torchsummary import summary
from tqdm import tqdm
import pandas as pd
import os
from PIL import Image
import matplotlib.image as mpimg
import warnings
warnings.filterwarnings('ignore')


# 訓練資料儲存路徑
path = 'output.txt'
f = open(path, 'w')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device:{device}")
print()

# 訓練參數設定
batch_size = 64
learning_rate = 1e-6
epochs = 30
early_stopping_tolerance = 3
early_stopping_threshold = 0.001
model_par_name = './model/30epoch_animal_128_re_par.pt'
model_name = './model/30epoch_animal_128_re.pt'

#資料夾路徑
data_dir = ".\Dataset_OpenCvDl_Hw2_Q5\dataset"

#training資料集
training_dir = os.path.join(data_dir,"training_dataset")

#validation資料集
validation_dir = os.path.join(data_dir,'validation_dataset')

transform_train = transforms.Compose(
    [transforms.Resize(128),
     transforms.CenterCrop(128),
     transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(),
     transforms.ToTensor(),
     transforms.RandomErasing(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
)

transform_val = transforms.Compose(
    [transforms.Resize(128),
     transforms.CenterCrop(128),
     transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
)

train_set = datasets.ImageFolder(training_dir,transform_train)
valid_set = datasets.ImageFolder(validation_dir,transform_val)

print(train_set.class_to_idx)

print(f"train size : {len(train_set)}, valid size : {len(valid_set)}")

train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_set,batch_size = batch_size,shuffle = True)

# train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size,shuffle = False)
# valid_loader = torch.utils.data.DataLoader(valid_set,batch_size = batch_size,shuffle = False)

# 建立模型
resNet = torchvision.models.resnet50(pretrained = True)
resNet.fc = nn.Sequential(
    nn.Linear(2048,1)
    # nn.Sigmoid()
)
resNet.to(device)

# loss function & optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(resNet.parameters(),lr=learning_rate,betas=[0.9,0.99])

print('-----------------------------')
print(f'criterion : {type(criterion).__name__}')
print(f'optimizer : {type (optimizer).__name__}')
print(f'Learning rate : {learning_rate}')
print(f'Batch size : {batch_size}, Epochs : {epochs}')
print('-----------------------------')
print()

def make_train_step(model,optimizer,loss_fn):
    def train_step(x,y):
        yhat = model(x)
        model.train()
        loss = loss_fn(yhat,y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
        return loss
    return train_step

train_step = make_train_step(resNet,optimizer,criterion)

# Training
print("Start training...")
start = time.time()

losses = []
val_losses = []

epoch_train_losses = []
epoch_val_losses = []


for epoch in range(epochs):
    epoch_loss = 0

    print(f'-------------------- epoch {epoch+1} --------------------')

    resNet.train()

    for i,data in tqdm(enumerate(train_loader),total=len(train_loader)):
        x_batch,y_batch = data
        x_batch = x_batch.to(device)

        y_batch = y_batch.unsqueeze(1).float()
        y_batch = y_batch.to(device)

        loss = train_step(x_batch,y_batch)
        epoch_loss += loss/len(train_loader)

        losses.append(loss)
        if i % 20 == 19:
            tqdm.write(
                f'[{epoch + 1}/{epochs}, {i + 1:3d}] loss: {loss} time:{(time.time()-start)/60:.2f}')
    
    epoch_train_losses.append(epoch_loss)
    print()
    print('\nEpoch : {}, train loss : {}'.format(epoch+1,epoch_loss))

    with torch.no_grad():
        cum_loss = 0
        
        for x_batch,y_batch in valid_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.unsqueeze(1).float()
            y_batch = y_batch.to(device)

            resNet.eval()

            yhat = resNet(x_batch)
            val_loss = criterion(yhat,y_batch)
            cum_loss += loss/len(valid_loader)
            val_losses.append(val_loss.item())

    epoch_val_losses.append(cum_loss)
    print('Epoch : {},val loss : {}'.format(epoch+1,cum_loss))

    best_loss = min(epoch_val_losses)

    if cum_loss <= best_loss:
        best_model_wts = resNet.state_dict()

    early_stopping_counter = 0

    if cum_loss > best_loss:
        early_stopping_counter += 1
    
    if(early_stopping_counter == early_stopping_tolerance) or (best_loss < early_stopping_threshold):
        print("/nTerminating : early stopping")
        break

resNet.load_state_dict(best_model_wts)

torch.save(resNet.state_dict(), model_par_name)
torch.save(resNet,model_name)


print("Finishing training!")

print()

