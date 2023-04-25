import os
import torch
import tqdm
import random
import numpy as np
import pandas as pd
import argparse

from dataset import AirDataset
from models import Linear, NLinear, DLinear
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Linear, NLinear, DLinear')
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--train_epochs', type=int, default=30)
parser.add_argument('--patient', type=int, default=5)
parser.add_argument('--ver', type=int, default=1)
args = parser.parse_args()

def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
fix_seed(0)

for i in ['공주', '노은동', '논산', '대천2동', '독곶리', '동문동', '모종동', '문창동', '성성동', 
          '신방동', '신흥동', '아름동', '예산군', '읍내동', '이원면', '정림동', '홍성읍']:
    data = i
    train_dataset = AirDataset(data, 'train')
    val_dataset = AirDataset(data, 'vali', is_train=False)
    test_dataset = AirDataset(data, 'test', is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, drop_last=True)

    model_dict = {
        'Linear': Linear,
        'NLinear': NLinear,
        'DLinear': DLinear
    }
    
    model = model_dict[args.model].Model(args).float().cuda()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    def eval(args, loader):
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for i, (x, y) in enumerate(loader):
                x = x.float().cuda()
                y = y.float().cuda()[:,:,0]
                outputs = model(x).cuda()
                loss = criterion(outputs, y)
                total_loss += loss.item()
        total_loss = total_loss/len(loader)
        return total_loss

    print(f"\n========================= Start Train for {i} =========================\n")
    model.train()
    best_loss = 100000000
    current_vali_loss = 100000000
    cnt = 0
    for epoch in range(args.train_epochs):
        train_loss = []
        for j, (x, y) in tqdm.tqdm(enumerate(train_loader), desc="train", total=len(train_loader)):
            optimizer.zero_grad()
            x = x.float().cuda()
            y = y.float().cuda()[:,:,0]

            outputs = model(x).cuda()
            loss = criterion(outputs, y)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
        train_loss = np.average(train_loss)
        vali_loss = eval(args, val_loader)
        test_loss = eval(args, test_loader)
        
        if best_loss > vali_loss:
            best_loss = vali_loss
            print(f'> > Saving model of {i}')
            torch.save(model, f'./checkpoints/best_model_{i}_{args.model}_{args.ver}.pt')
        
        if current_vali_loss > vali_loss:
           current_vali_loss = vali_loss
           cnt = 0
        else:
            cnt += 1
            print(f"> > Early Stopping | {cnt} / {args.patient}")            
        if cnt == args.patient:
            print("> > Early Stopping !\n")
            break
        
        print(f'Epoch: {epoch+1} | Train Loss:{train_loss:.5f} Vali Loss: {vali_loss:.5f} Test Loss: {test_loss:.5f}\n')
    
    
    print("=========================================================================\n")