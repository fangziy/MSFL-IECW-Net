import torch.nn as nn
import torch
from torchinfo import summary
import torch.nn.functional as F

class StarNet(nn.Module):

    def __init__(self, 
                num_label=1,
                mode="raw",
                ):
        super(StarNet, self).__init__()

        self.mode = mode

        if self.mode == "pre-RNN":
            self.rnn = nn.RNN(
                input_size=360,
                hidden_size=360,
                num_layers=1,
                batch_first=True,
            )
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,  # (-1, 1, 7200)
                out_channels=4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),          # (-1, 4, 7200)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(          # (-1, 4, 7200)
                in_channels=4,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),          # (-1, 16, 7200)
            nn.MaxPool1d(
                kernel_size=4,
            )                   # (-1, 16, 1800)
        )


        if self.mode == "post-RNN":
            self.rnn = nn.RNN(
                input_size=360,
                hidden_size=360,
                num_layers=1,
                batch_first=True,
            )

        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=15328,
                out_features=256,
            ),
            nn.ReLU(),   
            nn.Dropout(p=0.3),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(
                in_features=256,
                out_features=128,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        self.mu = nn.Linear(
            in_features=128,
            out_features=num_label,
        )
        self.sigma = nn.Sequential(
            nn.Linear(
                in_features=128,
                out_features=num_label,
            ),
            nn.Softplus()
        )

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
        x = self.fc2(x)

        return self.mu(x)

    def get_loss(self, y_true, y_pred, y_sigma):

        # return (torch.log(y_sigma)/2+ (y_true-y_pred)**2/(2*y_sigma)).mean() + 5
        return F.mse_loss(y_pred, y_true)



if __name__ == "__main__":
    net = StarNet(3)
    x = torch.randn(32, 3450)
    y = net(x)
    print(y[0].shape, y[1].shape)
    summary(net, (32, 3450))