import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1,:].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:,:].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0,2,1)).permute(0,2,1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = 48
        self.pred_len = 72
        self.ver = configs.ver

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.Seasonal_Linear = nn.Linear(self.seq_len, self.pred_len)
        self.Trend_Linear = nn.Linear(self.seq_len, self.pred_len)
        
        self.seasonal_pm_Linear = nn.Linear(self.seq_len, self.seq_len)
        self.trend_pm_Linear = nn.Linear(self.seq_len, self.seq_len)
        
        if self.ver == 1:
            self.sigmoid = nn.Sigmoid()
            # self.conv = nn.Conv1d(in_channels=self.time_channel, out_channels=1, kernel_size=1, bias=False)            
            self.temp_linear = nn.Linear(self.seq_len, self.seq_len)
            self.dir_linear = nn.Linear(self.seq_len, self.seq_len)
            self.vel_linear = nn.Linear(self.seq_len, self.seq_len)
            self.rain_linear = nn.Linear(self.seq_len, self.seq_len)
            self.humi_linear = nn.Linear(self.seq_len, self.seq_len)    
        elif self.ver == 2:
            self.conv = nn.Conv1d(in_channels=5, out_channels=1, kernel_size=1, bias=False)            
            self.linear = nn.Linear(self.seq_len, self.seq_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        pm = x[:,:,0].unsqueeze(2)
        seasonal_init, trend_init = self.decompsition(pm)
        seasonal_init = seasonal_init.squeeze()
        trend_init = trend_init.squeeze()
        seasonal_init = self.seasonal_pm_Linear(seasonal_init)
        trend_init = self.trend_pm_Linear(trend_init)

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
            seasonal_init = seasonal_init * temp * dir * vel * rain * humi
            trend_init = trend_init * temp * dir * vel * rain * humi

        elif self.ver == 2:
            x_mark = x[:,:,1:]
            feature = self.conv(x_mark.permute(0,2,1)).permute(0,2,1)
            feature = self.linear(feature.permute(0,2,1)).permute(0,2,1)
            feature = feature.squeeze()
            seasonal_init = seasonal_init*feature
            trend_init = trend_init*feature
            
        seasonal_output = self.Seasonal_Linear(seasonal_init)
        trend_output = self.Trend_Linear(trend_init)
        
        x = seasonal_output + trend_output
        return x # to [Batch, Output length, Channel]
