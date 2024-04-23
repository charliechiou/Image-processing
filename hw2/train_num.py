import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch import tensor
import time
import matplotlib.pyplot as plt
import random
import torch.utils.data as data

# 訓練資料儲存路徑
path = 'output.txt'
f = open(path, 'w')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device:{device}")
print()

# 訓練參數設定
batch_size = 64
learning_rate = 1e-6
epochs = 80
model_par_name = './model/epochs_80_par.pt'
model_name = './model/epochs_80.pt'

# transform = transforms.Compose(
#     [transforms.Resize(32),
#      transforms.RandomHorizontalFlip(),
#      transforms.RandomVerticalFlip(),
#      transforms.RandomRotation(30),
#      transforms.ToTensor(),
#      transforms.Normalize((0.5,), (0.5,))]
# )

transform = transforms.Compose(
    [transforms.Resize(32),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))]
)

# 資料集下載
data_set = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

print(f"total data size:{len(data_set)}")

train_percent = 0.8
val_percent = 0.2

data_num = len(data_set)

train_size = int(data_num * train_percent)
val_size = int(data_num * val_percent)

train_set, valid_set = data.random_split(data_set, [train_size, val_size])

print(f"train size : {len(train_set)}, valid size : {len(valid_set)}")

print()

# DataLoader建立
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
)

valid_loader = torch.utils.data.DataLoader(
    valid_set,
    batch_size=batch_size,
    shuffle=True,
)

# 分類名稱
classes = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')

# 建立模型
vgg19 = torchvision.models.vgg19_bn(num_classes=10)
vgg19.features[0] = nn.Conv2d(1,64,kernel_size=3,padding=1)
vgg19.classifier[6] = torch.nn.Linear(4096, 10)

vgg19.to(device)

# loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg19.parameters(),lr=learning_rate,betas=[0.9,0.99])
# optimizer = optim.SGD(vgg19.parameters(),lr=learning_rate)

print('-----------------------------')
print(f'criterion : {type(criterion).__name__}')
print(f'optimizer : {type (optimizer).__name__}')
print(f'Learning rate : {learning_rate}')
print(f'Batch size : {batch_size}, Epochs : {epochs}')
print('-----------------------------')
print()

# Training
print("Start training...")
start = time.time()
training_acc = []
training_loss = []
validation_acc = []
validation_loss = []
for epoch in range(epochs):
    print(f'-------------------- epoch {epoch+1} --------------------')

    vgg19.train()

    # 每個epochs歸零
    total = 0
    total_val = 0
    running_loss = 0.0
    train_loss = 0.0
    train_correct = 0
    val_correct = 0
    val_loss = 0.0

    # 計算每個batch
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        # 將gradient歸零
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = vgg19(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # batch loss計算
        running_loss += loss.item()

        # epoch loss計算
        train_loss += outputs.shape[0] * loss.item()

        # epoch accuracy計算
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        # 每50筆batch輸出訓練進度並將batch loss歸零
        if i % 50 == 49:
            print(
                f'[{epoch + 1}/{epochs}, {i + 1:3d}] loss: {running_loss / 50:.3f} time:{(time.time()-start)/60:.2f}')
            running_loss = 0.0

    # 輸出每個epoch loss及accuracy
    print(
        f"[{epoch + 1}/{epochs}] Training Accuracy:{100*train_correct/total:.3f} % Training Loss:{train_loss/len(train_set):.3f} time:{(time.time()-start)/60:.2f}")

    # 將train accuracy及loss加入array中用於後續繪圖
    training_acc.append(100*train_correct/total)
    training_loss.append(train_loss/len(train_set))

    # 儲存在訓練日誌中
    f.write(f"[Epochs:{epoch + 1}/{epochs}] Accuracy:{100*train_correct/total} % Loss:{train_loss/len(train_set):.3f} time:{(time.time()-start)/60:.2f}\n")

    # validation
    vgg19.eval()
    with torch.no_grad():
        for data in valid_loader:
            inputs_val, labels_val = data[0].to(device), data[1].to(device)

            outputs_val = vgg19(inputs_val)

            # 計算loss
            loss_val = criterion(outputs_val, labels_val)
            val_loss += outputs_val.shape[0] * loss_val.item()
            total_val += labels_val.size(0)

            # 計算accuracy
            _, predicted_val = torch.max(outputs_val.data, 1)
            val_correct += (predicted_val == labels_val).sum().item()

        # 輸出驗證數據
        print(
            f"[{epoch + 1}/{epochs}] Validation Accuracy:{100*val_correct/total_val:.3f} % Validation Loss:{val_loss/len(valid_set):.3f} time:{(time.time()-start)/60:.2f}")

        # 將驗證數據紀錄於訓練日誌中
        f.write(
            f"[Epochs:{epoch + 1}/{epochs}] Accuracy:{100*val_correct/total_val} % Loss:{val_loss/len(valid_set):.3f} time:{(time.time()-start)/60:.2f}\n")

        # 將train accuracy及loss加入array中用於後續繪圖
        validation_acc.append(100*val_correct/total_val)
        validation_loss.append(val_loss/len(valid_set))

print("Finishing training!")

# 儲存模型
torch.save(vgg19.state_dict(), model_par_name)
torch.save(vgg19,model_name)

# 繪出loss及accuracy圖形
plt.figure("loss")
plt.plot(validation_loss, '-', label='val loss')
plt.plot(training_loss, '-', label='train loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(
    loc='best',
    fontsize=10)
plt.title('Loss per epochs')
plt.savefig('loss_png')


plt.figure("accuracy")
plt.plot(validation_acc, '-', label='val acc')
plt.plot(training_acc, '-', label='train acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(
    loc='best',
    fontsize=10)
plt.title('Acc per epochs')
plt.savefig('acc_png')
# plt.show()


# testing

# 加載模型
vgg19_test = torchvision.models.vgg19_bn(num_classes=10)
vgg19_test.features[0] = nn.Conv2d(1,64,kernel_size=3,padding=1)
vgg19_test.to('cpu')

dataiter = iter(valid_loader)
images, labels = next(dataiter)
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

vgg19_test.load_state_dict(torch.load(model_par_name))

outputs = vgg19_test(images)
# print(outputs)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in valid_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = vgg19_test(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(
    f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in valid_loader:
        images, labels = data
        outputs = vgg19_test(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
