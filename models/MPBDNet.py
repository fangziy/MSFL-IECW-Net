'''
copyright
author: Ziyu Fang
email:fangziyushiwo.com
Date: 2024-12-25
版权所有，转载请注明作者和邮箱
'''

import torch
import torch.nn as nn
import numpy as np

class MPBDBlock(nn.Module):
    """
    @input_channel: Size of input channels
    @output_channel: Size of output channels
    """
    def __init__(self,
                 input_channel=1,
                 output_channel=4,
                 ):
        super(MPBDBlock, self).__init__()

        self.Block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=output_channel,
                out_channels=output_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.ReLU(),
       )
        self.Block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=True,
            ),
            nn.ReLU(),
            
            nn.Conv1d(
                in_channels=output_channel,
                out_channels=output_channel,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=True,
            ),
            nn.ReLU(),

       )
        
        self.downsample = nn.Sequential()
        if input_channel != output_channel:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                ),
            )
    def forward(self, x):

        x=self.Block1(x)+self.Block2(x)+self.downsample(x)
        # x=self.Block1(x)+self.downsample(x)
        return x

class embedding(nn.Module):
    def __init__(self,
                
                input_channel=1,
                embedding_c=50,
                kernel_size=3,
                overlap=1,
                padding=1,

                 ):
        super(embedding, self).__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channel,
                out_channels=embedding_c,
                kernel_size=kernel_size,
                stride=kernel_size-overlap,
                padding=padding,

                bias=True,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x=self.embedding(x)
        x=x.permute(0,2,1)
        return x

class MPBDNet(nn.Module):
    
    def __init__(self,
                num_label=3,
                list_inplanes= [3,6,18],
                num_rnn_sequence = 18,
                embedding_c=50,
                len_spectrum=3834,
                cls_columns=[],
                reg_columns=[],
                object_type="cls",
                ):
        super(MPBDNet, self).__init__()


        assert len(list_inplanes) < 6
        # assert len_spectrum % num_rnn_sequence == 0

        self.len_spectrum = len_spectrum
        self.num_rnn_sequence = num_rnn_sequence
        self.list_inplanes = list_inplanes.copy()
        self.list_inplanes.insert(0, 1)
        self.embedding_c=embedding_c
        self.object_type=object_type


        self.MPBDBlock_list = []
        for i in range(len(self.list_inplanes)-1):
            self.MPBDBlock_list.append(
                nn.Sequential(
                    MPBDBlock(self.list_inplanes[i], self.list_inplanes[i+1]),
                    MPBDBlock(self.list_inplanes[i+1], self.list_inplanes[i+1]),
                    MPBDBlock(self.list_inplanes[i+1], self.list_inplanes[i+1]),
                    nn.AvgPool1d(3),
                )
            )
        self.MPBDBlock_list = nn.Sequential(*self.MPBDBlock_list)

        self.embeding=embedding(
            input_channel=self.list_inplanes[-1],
            embedding_c=embedding_c,
            kernel_size=3,
            overlap=1,
            padding=1,
        )
        
        self.rnn = nn.LSTM(
            input_size=self.embedding_c,
            hidden_size=self.embedding_c,
            num_layers=1,
            batch_first=True,
        )
        self.rnn2 = nn.LSTM(
            input_size=self.embedding_c,
            hidden_size=self.embedding_c,
            num_layers=1,
            batch_first=True,
        )

        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=1500,
                out_features=256,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        if self.object_type=="cls" or self.object_type=="reg_cls":
            self.cls_benck = nn.Sequential(
                nn.Linear(
                    in_features=256,
                    out_features=128,
                ),
                nn.ReLU(),
                nn.Dropout(p=0.3),
            )
        
            self.cls = nn.Linear(
                in_features=128,
                out_features=num_label,
            )
        if self.object_type=="reg" or self.object_type=="reg_cls":
            self.reg_benck = nn.Sequential(
                nn.Linear(
                    in_features=256,
                    out_features=128,
                ),
                nn.ReLU(),
                nn.Dropout(p=0.3),
            )
            self.reg = nn.Linear(
                in_features=128,
                out_features=len(reg_columns),
            )

    def forward(self, x):
        if x.size(1)%8!=0:
            x=torch.cat([x, torch.zeros([x.size(0),x.size(1)%8]).to(x.device)], dim=1)
        B,L = x.size()
        x = x.reshape(-1, 1, L)
        x = self.MPBDBlock_list(x)
        x = torch.relu(x)
        x = self.embeding(x)
        # x=torch.concat(self.rnn(x)[0],torch.flip(self.rnn2(torch.flip(x,dims=[1]))[0],dims=[1]),dim=1)
        x=( self.rnn(x)[0]+torch.flip(self.rnn2(torch.flip(x,dims=[1]))[0],dims=[1]))/2
        x = self.fc1(x.flatten(1))
    

        if self.object_type=="cls":
            x=self.cls_benck(x)
            return self.cls(x)
        if self.object_type=="reg":
            x=self.reg_benck(x)
            return self.reg(x)
        
        if self.object_type=="reg_cls":
            x1=self.cls_benck(x)
            x2=self.reg_benck(x)
            x2=x1+x2
            cls=self.cls(x1)
            reg=self.reg(x2)
            x=torch.cat([cls,reg],dim=1)
            return x


