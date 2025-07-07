import torch.nn as nn
import torch
from torchinfo import summary
import torch.nn.functional as F

class StarNet(nn.Module):

    def __init__(self, 
                object_type='cls',
                num_cls=3,
                num_reg=5,
                mode="raw",
                len_spectrum=3834,
                ):
        super(StarNet, self).__init__()

        self.object_type = object_type
        self.mode = mode
        self.len_spectrum = len_spectrum

        if self.mode == "pre-RNN":
            self.rnn = nn.RNN(
                input_size=360,
                hidden_size=360,
                num_layers=1,
                batch_first=True,
            )
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,  # (-1, 1, len_spectrum)
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),          # (-1, 4, len_spectrum)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(          # (-1, 4, len_spectrum)
                in_channels=4,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),          # (-1, 16, len_spectrum)
            nn.MaxPool1d(
                kernel_size=4,
            )                   # (-1, 16, len_spectrum/4)
        )


        if self.mode == "post-RNN":
            self.rnn = nn.RNN(
                input_size=360,
                hidden_size=360,
                num_layers=1,
                batch_first=True,
            )

        # 动态计算fc1的输入维度
        fc1_input_features = self._calculate_fc1_input_features(len_spectrum)
        
        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=fc1_input_features,
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

    def _calculate_fc1_input_features(self, len_spectrum):
        """动态计算fc1层的输入特征数"""
        # 模拟前向传播过程
        with torch.no_grad():
            x = torch.randn(1, 1, len_spectrum)
            x = self.conv1(x)
            x = self.conv2(x)
            return x.flatten(1).size(1)

    def forward(self, x):
        B, L = x.size()
        if self.mode == "pre-RNN":
            x, h_n = self.rnn(x.view(-1,20,360))
            del h_n

        x = self.conv1(x.reshape(-1, 1, L))
        x = self.conv2(x)                       # -1, 16, 1800

        if self.mode == "post-RNN":
            x, h_n = self.rnn(x.permute(0, 2, 1).reshape(B,-1,360))
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

        # return (torch.log(y_sigma)/2+ (y_true-y_pred)**2/(2*y_sigma)).mean() + 5
        return F.mse_loss(y_pred, y_true)



if __name__ == "__main__":
    net = StarNet(object_type='cls', num_cls=3, len_spectrum=3450)
    x = torch.randn(32, 3450)
    y = net(x)
    print(y)
    summary(net, (32, 3450))