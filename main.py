import os
import random
import timeit

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import CONFIG
from CONFIG import model_config as mofig
from Model import ViT
from SRData import SRData


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    setSeed(mofig.random_seed)
    model = ViT().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                 betas=mofig.adam_betas, 
                                 lr=mofig.learning_rate,
                                 weight_decay=mofig.adam_weight_decay)
    train_loader, val_loader, test_loader = getDataLoaders()
    trainTest(model, criterion, optimizer, train_loader, val_loader)
    #train(model, criterion, optimizer, train_loader, val_loader)

def getDataLoaders():
    dataset_list = open(CONFIG.dataset_list).read()
    dataset_list = dataset_list.split(',')
    dataset = SRData(dataset_list)
    dataset_len = len(dataset_list)
    train_split = int(dataset_len * 0.8)
    val_split = int(dataset_len * 0.1)
    test_split = dataset_len - train_split - val_split

    train_set, val_set, test_set = random_split(dataset, 
                                                         [train_split, val_split, test_split])
    train_loader = DataLoader(train_set, batch_size=mofig.batch_size, shuffle=True)
    val_loader = DataLoader(train_set, batch_size=mofig.batch_size, shuffle=True)
    test_loader = DataLoader(train_set, batch_size=mofig.batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

def trainTest(model, criterion, optimizer, train_loader, val_loader):
    for idx, data in enumerate(tqdm(train_loader)):
        print(len(data[1]))
        if idx == 3:
            break

def train(model, criterion, optimizer, train_loader, val_loader):
    start = timeit.default_timer()
    for epoch in tqdm(range(mofig.epochs)):
        model.train()
        train_labels =[]
        train_preds = []
        train_running_loss = 0
        for idx, data in enumerate(tqdm(train_loader)):
            img = data[0].float().to(device)
            label = data[1].to(device)
            
            y = model(img)
            y_label = torch.argmax(y, dim=1)

            train_labels.extend(label.cpu().detach())
            train_preds.extend(y_label.cpu().detach())

            loss = criterion(img, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item()

        if epoch == 5:
            torch.save_to_dict(model, CONFIG.checkpoint)

        train_loss = train_running_loss / (idx + 1)

        model.eval()
        val_labels = []
        val_pred = []
        val_running_loss = 0
        with torch.no_grad():
            for idx, (img, label) in enumerate(tqdm(val_loader)):
                img = img.float().to(device)
                label = label.to(device)
                y = model(img)
                y_label = torch.argmax(y, dim=1)

                val_labels.extend(label.cpu().detach())
                val_pred.extend(y_label.cpu().detach())

                loss = criterion(y_label, label)
                val_running_loss += loss.item()
        
        val_loss = val_running_loss / (idx + 1)
        
        print("-"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print(f"Train Accuracy EPOCH {epoch+1}: {sum(1 for x,y in zip(train_preds, train_labels) if x == y) / len(train_labels):.4f}")
        print(f"Valid Accuracy EPOCH {epoch+1}: {sum(1 for x,y in zip(val_preds, val_labels) if x == y) / len(val_labels):.4f}")
        print("-"*30)
    
    stop = timeit.default_timer()
    print(f'Training time: {stop - start: .2f}sec')
   
def test(model, test_loader):
    labels = []
    imgs = []
    model.eval()
    with torch.no_grad():
        for idx, (img, label) in enumerage(tqdm(test_loader)):
            img = img.to(device)
            label = label.to(device)
            
            outputs = model(img)

            imgs.extend(img.detach().cpu())
            labels.extend([int(i) for i in torch.argmax(outputs, dim=1)])

def setSeed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()


