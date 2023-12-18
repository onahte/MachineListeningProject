import os
import random
import timeit

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import MulticlassConfusionMatrix as ConfusionMatrix
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score

import CONFIG
from CONFIG import model_config as mofig
from Model import ViT
from SRData import SRData


device = CONFIG.device

def main():
    setSeed(mofig.random_seed)
    model = ViT().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                 betas=mofig.adam_betas, 
                                 lr=mofig.learning_rate,
                                 weight_decay=mofig.adam_weight_decay)
    train_loader, val_loader, test_loader = getDataLoaders()
    #trainTest(model, criterion, optimizer, train_loader, val_loader)
    with torch.autograd.set_detect_anomaly(True):
        train(model, criterion, optimizer, train_loader, val_loader)

    test(model, criterion, optimizer, train_loader, val_loader)
    
    print(f'Model parameter count: {getParameterCount(model)}')


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
        train_roc_scores = []
        train_EERs = []
        train_running_loss = 0
        for idx, data in enumerate(tqdm(train_loader)):
            img = data[0].float().to(device)
            label = data[1].to(device)
            
            output = model(img)
            pred = torch.argmax(output, dim=1)

            train_labels.extend(label.cpu().detach())
            train_preds.extend(pred.cpu().detach())

            print(f'Output shape: {output.shape}')
            print(f'Label shape: {label.shape}')
            print(f'Predicted shape: {pred.shape}')

            loss = criterion(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_running_loss += loss.item()

            #train_roc_score = getROCScore(output, label)
            #train_roc_scores.extend(train_roc_score.cpu().detach())

            train_EER, _ = getEER(label, pred)
            train_EERs.extend(train_EER)

        if epoch == 5:
            torch.save_to_dict(model, CONFIG.checkpoint)

        train_loss = train_running_loss / (idx + 1)
        
        model.eval()
        val_labels = []
        val_pred = []
        val_roc_score = []
        val_EERs = []
        val_running_loss = 0
        with torch.no_grad():
            for idx, (img, label) in enumerate(tqdm(val_loader)):
                img = img.float().to(device)
                label = label.to(device)
                
                output = model(img)
                pred = torch.argmax(output, dim=1)

                val_labels.extend(label.cpu().detach())
                val_pred.extend(pred.cpu().detach())

                loss = criterion(output, label)
                val_running_loss += loss.item()
       
                #val_roc_score = getROCScore(output, label)
                #val_roc_scores.extend(val_roc_score.cpu().detach())

                val_EER, _ = getEER(label, pred)
                val_EERs.extend(val_EER)

        val_loss = val_running_loss / (idx + 1)
        print("-"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print(f"Train Accuracy EPOCH {epoch+1}: " 
                f"{sum(1 for x,y in zip(train_preds, train_labels) if x == y) / len(train_labels):.4f}")
        print(f"Valid Accuracy EPOCH {epoch+1}: "
                f"{sum(1 for x,y in zip(val_preds, val_labels) if x == y) / len(val_labels):.4f}")
        #print(f"Train ROC Score {epoch+1}: {sum(train_roc_scores) / len(train_roc_scores)}")
        #print(f"Valid ROC Score {epoch+1}: {sum(val_roc_scores) / len(val_roc_scores)}")
        print(f"EER Score: {val_sum(EERs) / len(val_EERs)}")
        print("-"*30)
    
    stop = timeit.default_timer()
    print(f'Training time: {stop - start: .2f}sec')

   
def test(model, test_loader):
    preds = []
    labels = []
    roc_scores = []
    EERs = []
    CM = []
    model.eval()
    with torch.no_grad():
        for idx, data in enumerage(tqdm(test_loader)):
            img = data[0].float().to(device)
            label = data[1].to(device)
            
            output = model(img)

            labels.extend(label.detach().cpu())
            preds.extend([int(i) for i in torch.argmax(output, dim=1)])
            #roc_scores.extend(getROCScore(output, label).cpu().detach())
            EER, confusion_mat = getEER(label, output)
            EERs.extend(EER)
            CM.append(confusion_mat)

    print("-"*30)
    print(f"Accuracy: {sum(1 for x, y in zip(preds, labels) if x == y) / len(labels):.4f}")
    #print(f"ROC Score: {sum(roc_scores) / len(roc_scores)}")
    print(f"EER Score: {sum(EERs) / len(EERs)}")
    print("-"*30)

    avg_conf_mat = computAVG(EERS)
    saveConfusionMatrixPlot(avg_conf_mat)


def getParameterCount(model):
    return sum(p.numel() for p in model.parameters())


def getROCScore(label, pred):
    prob_estimate = F.softmax(pred, dim=1).detach()
    return roc_auc_score(label, prob_estimate, multi_class="ovr")
             

def getEER(labels, pred):
    CM = ConfusionMatrix(mofig.num_classes)
    confusion_mat = CM(pred, labels)
    #print(confusion_mat)
    EER = []
    for n in range(mofig.num_classes):
        true_pos = confusion_mat[n][n]
        false_pos = sum([i[n] for i in confusion_mat]) - true_pos
        false_neg = sum(confusion_mat[n]) - true_pos
        true_neg = sum([sum(i) for i in confusion_mat]) - true_pos - false_pos - false_neg

        FAR = false_pos / (false_pos + true_neg) * 100
        FRR = false_neg / (false_neg + true_pos) * 100

        EER.append((FAR + FRR) / 2)

    return EER, confusion_mat


def computAVG(matrices):
    count = len(matrices)
    i_len = len(len(matrices[0]))
    j_len = len(len(matrices[0][0]))
    avg_matrix = np.zeros((i,j))
    for i in range(i_len):
        for j in range(j_len):
            pos_sum = 0
            for matrix in matrices:
                pos_sum += matrix[i][j]
            avg_matrix[i][j] = pos_sum / count
    return avg_matrix


def saveConfusionMatrixPlot(matrix): 
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.title('Confusion Matrix')
    classes = ['classes' + i for i in range(mofig.num_classes)]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f"{matrix[i, j]}", ha='center', va='center', color='black')

    plt.savefig(mofig.confusion_mat, format='png')


def setSeed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()


