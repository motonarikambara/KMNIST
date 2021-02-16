import os
import random

import numpy as np
import pandas as pd
import cv2


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD,Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet50
from tqdm import tqdm
from TransformerModel import TransformerModel

# def seed_everything(seed=42):
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True

class KMNISTDataset(Dataset):
    def __init__(self, fname_list, label_list, image_dir, transform=None):
        super().__init__()
        self.fname_list = fname_list
        self.label_list = label_list
        self.image_dir = image_dir
        self.transform = transform
        
    # Datasetを実装するときにはtorch.utils.data.Datasetを継承する
    # __len__と__getitem__を実装する
    
    def __len__(self):
        return len(self.fname_list)
    
    def __getitem__(self, idx):
        fname = self.fname_list[idx]
        label = self.label_list[idx]
        
        image = cv2.imread(os.path.join(self.image_dir, fname))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.transform is not None:
            image = self.transform(image)
        # __getitem__でデータを返す前にtransformでデータに前処理をしてから返すことがポイント
        return image, label

# class MLP(nn.Module):
#     def __init__(self, input_dim=28*28, hidden_dim=128, output_dim=10):
#         super().__init__()
#         # nn.Linearは fully-connected layer (全結合層)のことです．
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#         self.activation = nn.ReLU()
    
#     def forward(self, x):
#         # 1次元のベクトルにする
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.activation(x)
#         x = self.fc2(x)
        
#         return x

class ImageFeatureExtractor1(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor1, self).__init__()
        self.net = resnet50(pretrained=True).to("cuda:0")
        self.cnn = nn.Sequential(
                                nn.Conv2d(256, 512, 1), 
                                nn.AvgPool2d(2,2), 
                                nn.Conv2d(512, 1024, 1), 
                                nn.AvgPool2d(2,2), 
                                nn.Conv2d(1024, 2048, 1), 
                                nn.AvgPool2d(2,2)).to("cuda:0")
        # freeze cnn
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False
    
    def forward(self, img):
        tmpimg = img.reshape([1, 3, 224, 224]).to("cuda:0")
        # tmpimg = torch.from_numpy(tmpimg.copy()).to("cuda:0")
        feature = self.net.conv1(tmpimg)
        feature = self.net.bn1(feature)
        feature = self.net.relu(feature)
        feature = self.net.maxpool(feature)
        feature = self.net.layer1(feature)
        feature = self.cnn(feature)
        feature = F.adaptive_avg_pool2d(feature, (1, 1))
        device2 = torch.device('cpu')
        feature = feature.to(device2)
        feature = feature.reshape([1, 2048]).detach().numpy()
        return feature

class ImageFeatureExtractor2(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor2, self).__init__()
        self.net = resnet50(pretrained=True).to("cuda:0")
        self.cnn = nn.Sequential(
                                nn.Conv2d(512, 1024, 1), 
                                nn.AvgPool2d(2,2), 
                                nn.Conv2d(1024, 2048, 1), 
                                nn.AvgPool2d(2,2)).to("cuda:0")
        # freeze cnn
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False
    
    def forward(self, img):
        tmpimg = img.reshape([1, 3, 224, 224]).to("cuda:0")
        # tmpimg = torch.from_numpy(tmpimg.copy()).to("cuda:0")
        feature = self.net.conv1(tmpimg)
        feature = self.net.bn1(feature)
        feature = self.net.relu(feature)
        feature = self.net.maxpool(feature)
        feature = self.net.layer1(feature)
        feature = self.net.layer2(feature)
        feature = self.cnn(feature)
        feature = F.adaptive_avg_pool2d(feature, (1, 1))
        device2 = torch.device('cpu')
        feature = feature.to(device2)
        feature = feature.reshape([1, 2048]).detach().numpy()
        return feature

class ImageFeatureExtractor3(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor3, self).__init__()
        cnn = resnet50(pretrained=True).to("cuda:0")
        self.net = nn.Sequential(*list(cnn.children())[:-1])
        # freeze cnn
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False
    
    def forward(self, img):
        tmpimg = img.reshape([1, 3, 224, 224]).to("cuda:0")
        # tmpimg = torch.from_numpy(tmpimg.copy()).to("cuda:0")
        feature = self.net(tmpimg)
        device2 = torch.device('cpu')
        feature = feature.to(device2)
        feature = feature.reshape([1, 2048]).detach().numpy()
        return feature

if __name__ == '__main__':
    INPUT_DIR = './'

    PATH = {
        'train': os.path.join(INPUT_DIR, 'train.csv'),
        'sample_submission': os.path.join(INPUT_DIR, 'sample_submission.csv'),
        'train_image_dir': os.path.join(INPUT_DIR, 'train_images/train_images'),
        'test_image_dir': os.path.join(INPUT_DIR, 'test_images/test_images'),
    }

    ID = 'fname'
    TARGET = 'label'

    # SEED = 42
    # seed_everything(SEED)

    # GPU settings for PyTorch (explained later...)
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Parameters for neural network. We will see the details later...
    PARAMS = {
        'valid_size': 0.2,
        'batch_size': 64,
        'epochs': 5,
        'lr': 0.001,
        'valid_batch_size': 256,
        'test_batch_size': 256,
    }

    train_df = pd.read_csv(PATH['train'])
    sample_submission_df = pd.read_csv(PATH['sample_submission'])


    train_df, valid_df = train_test_split(
        train_df, test_size=PARAMS['valid_size'], shuffle=True
    )
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        # numpy.arrayで読み込まれた画像をPyTorch用のTensorに変換します．
        transforms.Normalize((0.5, ), (0.5, ))
        #正規化の処理も加えます。
    ])

    model = TransformerModel()

    train_dataset = KMNISTDataset(train_df[ID], train_df[TARGET], PATH['train_image_dir'], transform=transform)
    valid_dataset = KMNISTDataset(valid_df[ID], valid_df[TARGET], PATH['train_image_dir'], transform=transform)

    # DataLoaderを用いてバッチサイズ分のデータを生成します。shuffleをtrueにすることでデータをshuffleしてくれます
    train_dataloader = DataLoader(train_dataset, batch_size=PARAMS['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=PARAMS['valid_batch_size'], shuffle=False)
    a = torch.tensor(0.9, requires_grad=True)
    b = torch.tensor(0.999, requires_grad=True)
    params = [a,b]
    optim = Adam(params, lr=PARAMS['lr'])
    criterion = nn.CrossEntropyLoss()
    feature1 = ImageFeatureExtractor1()
    feature2 = ImageFeatureExtractor2()
    feature3 = ImageFeatureExtractor3()


    def accuracy_score_torch(y_pred, y):
        y_pred = torch.argmax(y_pred, axis=1).cpu().numpy()
        y = y.cpu().numpy()

        return accuracy_score(y_pred, y)

    for epoch in range(PARAMS['epochs']):
        # epochループを回す
        model.train()
        train_loss_list = []
        train_accuracy_list = []
        
        for x, y in tqdm(train_dataloader):
            print(x.shape)
            # 先ほど定義したdataloaderから画像とラベルのセットのdataを取得
            # x = x.to(dtype=torch.float32)
            y = y.to(dtype=torch.long)
            x1 = feature1(x)
            x2 = feature2(x)
            x3 = feature3(x)
            tmpx = np.append(x1, x2, axis=0)
            newx = np.append(tmpx, x3, axis=0)
            input = torch.tensor(newx, dtype=torch.float32).to("cuda:0")

            
            # pytorchでは通常誤差逆伝播を行う前に毎回勾配をゼロにする必要がある
            optim.zero_grad()
            # 順伝播を行う
            y_pred = model(input)
            # lossの定義 今回はcross entropyを用います
            loss = criterion(y_pred, y)
            # 誤差逆伝播を行なってモデルを修正します(誤差逆伝播についてはhttp://hokuts.com/2016/05/29/bp1/)
            loss.backward() # 逆伝播の計算
            # 逆伝播の結果からモデルを更新
            optim.step()
            
            train_loss_list.append(loss.item())
            train_accuracy_list.append(accuracy_score_torch(y_pred, y))
        
        model.eval()
        valid_loss_list = []
        valid_accuracy_list = []

        for x, y in tqdm(valid_dataloader):
            # x = x.to(dtype=torch.float32).to("cuda:0")
            y = y.to(dtype=torch.long).to("cuda:0")
            x1 = feature1(x)
            x2 = feature2(x)
            x3 = feature3(x)
            tmpx = np.append(x1, x2, axis=0)
            newx = np.append(tmpx, x3, axis=0)
            input = torch.tensor(newx, dtype=torch.float32).to("cuda:0")
            
            with torch.no_grad():
                y_pred = model(input)
                loss = criterion(y_pred, y)
            
            valid_loss_list.append(loss.item())
            valid_accuracy_list.append(accuracy_score_torch(y_pred, y))
        
        print('epoch: {}/{} - loss: {:.5f} - accuracy: {:.3f} - val_loss: {:.5f} - val_accuracy: {:.3f}'.format(
            epoch,
            PARAMS['epochs'], 
            np.mean(train_loss_list),
            np.mean(train_accuracy_list),
            np.mean(valid_loss_list),
            np.mean(valid_accuracy_list)
        ))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    test_dataset = KMNISTDataset(
        sample_submission_df[ID],
        sample_submission_df[TARGET],
        PATH['test_image_dir'],
        transform=transform
    )

    test_dataloader = DataLoader(test_dataset, batch_size=PARAMS['test_batch_size'], shuffle=False)

    predictions = []

    # テスト
    # for x, _ in test_dataloader:
    #     x = x.to(dtype=torch.float32).to("cuda:0")
        
    #     with torch.no_grad():
    #         y_pred = model.sample(x)
    #         y_pred = torch.argmax(y_pred, axis=1).cpu().numpy()
    #         y_pred = y_pred.tolist()
            
    #     predictions += y_pred

    # sample_submission_df[TARGET] = predictions

    # sample_submission_df.to_csv('submission.csv', index=False)

