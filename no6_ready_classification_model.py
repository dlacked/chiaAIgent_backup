import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import numpy as np
import datetime
from torch.utils.data import random_split

class ToothClassification(nn.Module):
    def __init__(self, num_classes=12):
        super(ToothClassification, self).__init__()

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # IMAGENET1K_V1으로 사물 인식 능력 빌려오기

        num_ftrs = self.resnet.fc.in_features #fc 전 층 차원
        
        self.resnet.fc = nn.Identity() # IMAGENET1K_V1으로 인한 분류를 막고 치아 번호 분류를 위함
        
        self.sincos_fc = nn.Sequential( # 각도 데이터를 위한 신경망
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        ) 

        self.common_fc = nn.Sequential(
            nn.Linear(num_ftrs + 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.head_tooth = nn.Linear(256, num_classes)
        self.head_cavity = nn.Linear(256, 2)

    def forward(self, img, theta):
        '''
        forward 함수는 PyTorch에서 데이터가 모델을 통과하며
        예측값을 계산하는 과정을 담당하는 프로토콜이다.
        '''
        img_features = self.resnet(img)

        sincos_features = self.sincos_fc(theta)

        combine_features = torch.cat((img_features, sincos_features), dim=1)
        x = self.common_fc(combine_features)

        out_tooth = self.head_tooth(x)
        out_cavity = self.head_cavity(x)

        return out_tooth, out_cavity

class ToothDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform, oral_type):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.oral_type = oral_type
        if self.oral_type == 'lower':
            self.label_map = {
                31:0, 32:1, 33:2, 34:3, 35:4, 36:5,
                41:6, 42:7, 43:8, 44:9, 45:10, 46:11
            }
        else:
            self.label_map = {
                11:0, 12:1, 13:2, 14:3, 15:4, 16:5,
                21:6, 22:7, 23:8, 24:9, 25:10, 26:11
            }

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.df.iloc[index]['img_dir']
        image = Image.open(img_path).convert('RGB')

        sin = float(self.df.iloc[index]['teeth_sin'])
        cos = float(self.df.iloc[index]['teeth_cos'])
        sincos = torch.tensor([sin, cos], dtype=torch.float32) #sin/cos 정규화, input이 2개가 됨

        tooth_num = self.label_map[int(self.df.iloc[index]['teeth_num'])]
        cavity_label = int(self.df.iloc[index]['is_cavity'])

        if self.transform:
            image = self.transform(image)
        
        return image, sincos, tooth_num, cavity_label

if __name__ == "__main__":
    '''
    일단 lower만 모델 훈련시켜보기
    '''
    
    print('lower/upper Default: lower')
    oral_type = input()
    if oral_type != 'upper':
        oral_type = 'lower'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = ToothDataset(
        csv_path=f'./cropped_dataset/train/{oral_type}/csv/metadata.csv', # lower 설정
        transform=train_transform,
        oral_type=oral_type
    )

    train_size = int(0.95 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

    model = ToothClassification(num_classes=12).to(device)

    criterion_tooth_num = nn.CrossEntropyLoss()
    criterion_cavity = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 12.0]).to(device))
    # 충치 이미지의 부족으로 충치 클래스에 가중치 부여
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(2): # epochs 수
        print(f"{epoch+1} Epochs Running")
        model.train()
        running_loss = 0

        for batch_index, (images, sincos, tooth_num_labels, cavity_labels) in enumerate(train_loader):
            images, sincos= images.to(device), sincos.to(device)
            tooth_num_labels, cavity_labels = tooth_num_labels.to(device), cavity_labels.to(device)

            optimizer.zero_grad()
            out_t, out_c = model(images, sincos)

            loss_t = criterion_tooth_num(out_t, tooth_num_labels)
            loss_c = criterion_cavity(out_c, cavity_labels)
            total_loss = loss_t + loss_c

            total_loss.backward()
            optimizer.step()

            if batch_index % 20 == 0:
                print(f'{datetime.datetime.now()}, {batch_index*64}')
                print(f'loss_teeth_num: {loss_t.item():.4f}')
                print(f'loss_cavity: {loss_c.item():.4f}')
            running_loss += total_loss.item()

        model.eval()
        tn_correct, c_correct = 0, 0
        total = 0

        with torch.no_grad():
            for images, sincos, tooth_num_labels, cavity_labels in val_loader:
                images, sincos, tooth_num_labels, cavity_labels = images.to(device), sincos.to(device), tooth_num_labels.to(device), cavity_labels.to(device)
                
                out_t, out_c = model(images, sincos)
                _, pred_t = torch.max(out_t, 1)
                _, pred_c = torch.max(out_c, 1)
                
                total += tooth_num_labels.size(0)
                tn_correct += (pred_t == tooth_num_labels).sum().item()
                c_correct += (pred_c == cavity_labels).sum().item()
        
        accuracy = 100 * tn_correct / total
        print(f"Epoch {epoch+1} - Tooth Acc: {100*tn_correct/total:.2f}% | Cavity Acc: {100*c_correct/total:.2f}%")

    if not os.path.exists(f'./runs/cls/{oral_type}'):
        os.makedirs(f'./runs/cls/{oral_type}')

    torch.save(model.state_dict(), f"runs/cls/{oral_type}/best.pth")
