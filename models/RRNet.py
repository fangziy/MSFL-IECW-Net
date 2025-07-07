import torch
import torch.nn as nn
import numpy as np



class ResBlock(nn.Module):
    """
    Residuals Module
    @input_channel: Size of input channels
    @output_channel: Size of output channels
    """

    def __init__(self,
                 input_channel=1,
                 output_channel=4,
                 ):
        super(ResBlock, self).__init__()

        self.ResBlock = nn.Sequential(
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
        return self.ResBlock(x) + self.downsample(x)


class embedding(nn.Module):
    def __init__(self,
                
                input_channel=1,
                embedding_c=50,
                kernel_size=3,
                overlap=0,
                padding=1,

                 ):
        super(embedding, self).__init__()
        self.embedding_c=embedding_c
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


class RRNet(nn.Module):
    """
    Residuals Module
    @mode:
        "raw": The raw ResNet model.
        "pre-RNN": Pre-embedding RNN on ResNet model.
        "post-RNN": Post-embedding RNN on ResNet model.
    @object_type: 'cls', 'reg', 'cls_reg', 'reg_cls'
    @num_cls: The number of classes for classification.
    @num_reg: The number of regression outputs.
    @list_inplanes: The number of channels per residual block in ResBlock list.
    @len_spectrum: The length of input spectrum.
    """

    def __init__(self,
                object_type='cls',
                mode="raw",
                num_cls=3,
                num_reg=5,
                list_inplanes= [3,6,18],
                len_spectrum=3834,
                sequence_len=50
                ):
        super(RRNet, self).__init__()

        assert len(list_inplanes) < 6
        # assert len_spectrum % num_rnn_sequence == 0

        self.object_type = object_type
        self.mode = mode
        self.len_spectrum = len_spectrum
        self.sequence_len = sequence_len
        self.list_inplanes = list_inplanes.copy()
        self.list_inplanes.insert(0, 1)
        self.times=int(np.ceil(len_spectrum/8))

        if self.mode == "pre-RNN":
            self.num_rnn_sequence = 20  # 设置默认值
            self.rnn = nn.RNN(
                input_size=self.len_spectrum // self.num_rnn_sequence,
                hidden_size=self.len_spectrum // self.num_rnn_sequence,
                num_layers=1,
                batch_first=True,
            )

        ResBlock_modules = []
        for i in range(len(self.list_inplanes)-1):
            ResBlock_modules.append(
                nn.Sequential(
                    ResBlock(self.list_inplanes[i], self.list_inplanes[i+1]),
                    nn.AvgPool1d(3),
                )
            )
        self.ResBlock_list = nn.Sequential(*ResBlock_modules)

        if self.mode == "post-RNN":
            self.embeding=embedding(input_channel=self.list_inplanes[-1], embedding_c=sequence_len, kernel_size=3, overlap=0, padding=1)
            self.rnn = nn.RNN(
                input_size=sequence_len,
                hidden_size=sequence_len,
                num_layers=1,
                batch_first=True,
            )

        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=1000,
                out_features=256,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        
        if self.object_type in ['cls', 'cls_reg', 'reg_cls']:
            self.cls_fc = nn.Sequential(
                nn.Linear(
                    in_features=256,
                    out_features=128,
                ),
                nn.ReLU(),
                nn.Dropout(p=0.3),
            )
            self.cls = nn.Linear(
                in_features=128,
                out_features=num_cls,
            )
            
        if self.object_type in ['reg', 'cls_reg', 'reg_cls']:
            self.reg_fc = nn.Sequential(
                nn.Linear(
                    in_features=256,
                    out_features=128,
                ),
                nn.ReLU(),
                nn.Dropout(p=0.3),
            )
            self.reg = nn.Linear(
                in_features=128,
                out_features=num_reg,
            )

    def forward(self, x):

        x=torch.cat([x, torch.zeros([x.size(0),x.size(1)%8]).to(x.device)], dim=1)
        B,L = x.size()

        if self.mode == "pre-RNN":
            x, h_n = self.rnn(x.view(B, self.num_rnn_sequence, -1))
            del h_n

        x = x.reshape(-1, 1, L)

        x = self.ResBlock_list(x)
        x = torch.relu(x)
        if self.mode == "post-RNN":
            x = self.embeding(x)
            x = x.permute(0, 2, 1).reshape(B, -1, self.sequence_len)
            x, h_n = self.rnn(x)
            del h_n

        x = self.fc1(x.flatten(1))
        
        out_put = {}
        
        if self.object_type in ['cls', 'cls_reg', 'reg_cls']:
            cls_output = self.cls(self.cls_fc(x))
            out_put['cls'] = cls_output
            
        if self.object_type in ['reg', 'cls_reg', 'reg_cls']:
            reg_output = self.reg(self.reg_fc(x))
            out_put['reg'] = reg_output
        
        return out_put

    def get_loss(self, y_true, y_pred, y_sigma):

        return (torch.log(y_sigma)/2+ (y_true-y_pred)**2/(2*y_sigma)).mean() + 5

