import torch
import pandas as pd
import random
import numpy as np
import os
import csv 
import tqdm

from dataset import PredDataset, SubsetRandomSampler
from models import Linear, NLinear, DLinear
from torch.utils.data import DataLoader

def fix_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
fix_seed(0)

region = ['공주', '노은동', '논산', '대천2동', '독곶리', '동문동', '모종동', '문창동','성성동', 
          '신방동', '신흥동', '아름동', '예산군', '읍내동', '이원면', '정림동', '홍성읍']
for i in tqdm.tqdm(region, desc='Region', total=len(region)):

    Pred_dataset = PredDataset(i)
    indices = [i for i in range(len(Pred_dataset)) if i%120==0]
    pred_loader = DataLoader(Pred_dataset, batch_size=32, shuffle=False, sampler=SubsetRandomSampler(indices))
    
    for m in ['Linear', 'NLinear', 'DLinear']:
        for v in [1,2]:
            model = torch.load(f'./checkpoints/best_model_{i}_{m}_{v}.pt').cuda()
            preds = []
            current = 0
            with torch.no_grad():
                for j, x in enumerate(pred_loader):
                    x = x.float().cuda()
                    outputs = model(x).cuda()
                    pred = outputs.detach().cpu().numpy()
                    preds.append(pred)
                preds = np.array(preds)
                preds = preds.reshape(-1, 1).squeeze()
            
            with open("./dataset/answer_sample.csv") as f:
                reader = csv.reader(f)
                content = []
                for row in reader:
                    if i=='공주' and row[0] == '연도':
                        row = ",".join(row)
                        content.append(row)
                    elif i!='공주' and row[0] == '연도':
                        continue
                    if row[0] != '연도' and row[-2] == i:
                        row[-1] = preds[current]
                        current += 1
                        row = ",".join(map(str, row))
                        content.append(row)
                        
            if i == '공주':
                f = open(f"./result/sample2_{m}_{v}.csv", "w", encoding='UTF-8')
                for row in content:
                    f.write(row)
                    f.write('\n')
                f.close()
            else:
                f = open(f"./result/sample2_{m}_{v}.csv", "a", encoding='UTF-8')
                for row in content:
                    f.write(row)
                    f.write('\n')
                f.close()
