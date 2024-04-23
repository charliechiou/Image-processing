from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

print('Start testing model')

class animalDataset(Dataset):
    def __init__(self,img_dir,transforms = None):
        self.img_dir = img_dir
        self.transform = transforms
        self.subfolders = [f for f in os.listdir(self.img_dir) if os.path.isdir(os.path.join(self.img_dir,f))]
        self.img_labels = []
        self.img = []

        for subfolder in self.subfolders:
            files = os.listdir(os.path.join(self.img_dir,subfolder))
            subfolder_name = os.path.join(img_dir,subfolder)
            for f in files:
                self.img_labels.append(subfolder)
                img_name = os.path.join(subfolder_name,f)
                self.img.append(img_name)

    def __len__(self):
        count = 0
        for subfolder in self.subfolders:
            files = os.listdir(os.path.join(self.img_dir,subfolder))
            count += len(files)
        return count
    
    def __getitem__(self,idx):
        image = self.img[idx]
        label = self.img_labels[idx]

        return image,label
    
dst = animalDataset("C:\chiou\opencv_class\hw2\Dataset_OpenCvDl_Hw2_Q5\dataset\\validation_dataset")

transform_animal = transforms.Compose(
    [transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor()])

#Original
resNet = torch.load(
    "C:\chiou\opencv_class\hw2\model\\animal\\30epoch_128\\30epoch_128.pt"
)

resNet.to('cpu')
resNet.eval()

correct = 0
total = 0

for i in range(len(dst)):
    try:
        img_class = Image.open(dst[i][0])
        img_tensor = transform_animal(img_class)

        sample = torch.unsqueeze(img_tensor,dim=0)

        with torch.no_grad():
            outputs = resNet(sample)
            outputs = torch.sigmoid(outputs)

        if outputs > 0.5 :
            ani = 'Dog'
        if outputs < 0.5 :
            ani = 'Cat'
        if ani == dst[i][1]:
            correct += 1

        total += 1
    except:
        print(f'error pic:{dst[i][0]}')

#With Random Erasing
resNet = torch.load(
    "C:\chiou\opencv_class\hw2\model\\animal\\30epoch_128_re\\30epoch_128_re.pt"
)

resNet.to('cpu')
resNet.eval()

correct_re = 0
total_re = 0

for i in range(len(dst)):
    try:
        img_class = Image.open(dst[i][0])
        img_tensor = transform_animal(img_class)

        sample = torch.unsqueeze(img_tensor,dim=0)

        with torch.no_grad():
            outputs = resNet(sample)
            outputs = torch.sigmoid(outputs)

        if outputs > 0.5 :
            ani = 'Dog'
        if outputs < 0.5 :
            ani = 'Cat'
        if ani == dst[i][1]:
            correct_re += 1

        total_re += 1
    except:
        print(f'error pic:{dst[i][0]}')

print(f'accuracy without random erasing:{correct/total:.4f}%')
print(f'accuracy with random erasing:{correct_re/total_re:.4f}%')

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, 1.005*y[i], round(y[i],1), ha = 'center')

x = ['Without Random erasing','With Random erasin']
y = [correct/total*100,correct_re/total_re*100]
plt.bar(x,y,width=0.5)
plt.title('Accuracy Comparison')
plt.ylabel('Accuracy (%)')
addlabels(x,y)
plt.show()