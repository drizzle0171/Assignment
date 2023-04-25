import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, config):
        super(Model, self).__init__()
        self.seq_len = 48
        self.pred_len = 72
        self.ver = config.ver
        
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.power_Linear = nn.Linear(self.seq_len, self.seq_len)
        
        if self.ver == 1:
            self.sigmoid = nn.Sigmoid()
            self.temp_linear = nn.Linear(self.seq_len, self.seq_len)
            self.dir_linear = nn.Linear(self.seq_len, self.seq_len)
            self.vel_linear = nn.Linear(self.seq_len, self.seq_len)
            self.rain_linear = nn.Linear(self.seq_len, self.seq_len)
            self.humi_linear = nn.Linear(self.seq_len, self.seq_len)
            
        elif self.ver == 2:
            self.conv = nn.Conv1d(in_channels=5, out_channels=1, kernel_size=1, bias=False)            
            self.linear = nn.Linear(self.seq_len, self.seq_len)
    
    def forward(self, x):
        
        pm = x[:,:,0]        
        pm = self.power_Linear(pm)
        
        if self.ver == 1:
            temp = x[:,:,1]
            dir = x[:,:,2]
            vel = x[:,:,3]
            rain = x[:,:,4]
            humi = x[:,:,5]

            temp = self.temp_linear(temp)
            temp = self.sigmoid(temp)
            dir = self.dir_linear(dir)
            dir = self.sigmoid(dir)
            vel = self.vel_linear(vel)
            vel = self.sigmoid(vel)
            rain = self.rain_linear(rain)
            rain = self.sigmoid(rain)
            humi = self.humi_linear(humi)
            humi = self.sigmoid(humi)
            pm = pm*temp*dir*vel*rain*humi 

        if self.ver == 2:    
            x_mark = x[:,:,1:]
            feature = self.conv(x_mark.permute(0,2,1)).permute(0,2,1)
            feature = self.linear(feature.permute(0,2,1)).permute(0,2,1)
            feature = feature.squeeze()
            pm = pm*feature
        
        x = self.Linear(pm)    

        return x # [Batch, Output length, Channel]