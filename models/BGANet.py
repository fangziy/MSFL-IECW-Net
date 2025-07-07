import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, batch_norm_size=None):
        super(GRUBlock, self).__init__()

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        # 只在第一层使用batch_norm
        if batch_norm_size is not None:
            self.batch_norm = nn.BatchNorm1d(batch_norm_size)
        else:
            self.batch_norm = None

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.dropout(x)

        if self.batch_norm is not None:
            x = x.permute(0, 2, 1)  # Adjust dimensions for batch norm
            x = self.batch_norm(x)
            x = x.permute(0, 2, 1)  # Re-adjust dimensions back

        return x



class BGANet(nn.Module):
    def __init__(self, 
                 object_type='reg',
                 n_subsequence = 6, 
                 len_spectrum=3456, 
                 list_GRU_hidden=[64, 32], 
                 dropout=0.2,
                 dropout_self=0.,
                 num_cls = 3,
                 num_reg=5,     
                 num_heads = 1):
        super(BGANet, self).__init__()

        # assert len_spectrum % n_subsequence == 0
        self.len_spectrum =  math.ceil(len_spectrum / n_subsequence)*n_subsequence
        self.raw_len = len_spectrum
        self.seq_length = int(self.len_spectrum/n_subsequence)  #每个子波段长度
        self.n_subsequence = n_subsequence
        self.object_type = object_type
        # 定义GRU层
        self.GRU_list = nn.ModuleList()
        # self.GRU_list = []
        for i in range(len(list_GRU_hidden)):
            gru_block_name = f'GRUBlock_{i}'
            if i == 0:
                self.GRU_list.append(GRUBlock(self.seq_length, list_GRU_hidden[i], dropout, batch_norm_size=2*list_GRU_hidden[i]))
            else:
                self.GRU_list.append(GRUBlock(2*list_GRU_hidden[i-1], list_GRU_hidden[i], dropout))

        # self.GRU_list = nn.Sequential(*self.GRU_list) 

        # 定义注意力机制
        self.attention = nn.MultiheadAttention(2 * list_GRU_hidden[-1], num_heads, batch_first=True,dropout=dropout_self)
        # self.attention= SeqSelfAttention(2 * list_GRU_hidden[-1])        

        # 定义输出层
        if object_type == 'reg' or object_type == 'reg_cls' or object_type == 'cls_reg':
            self.mu = nn.Sequential(
                nn.Linear(
                    in_features=2 * list_GRU_hidden[-1],
                    out_features=num_reg,
                ),
            )
            self.sigma = nn.Sequential(
                nn.Linear(
                    in_features=2 * list_GRU_hidden[-1],
                    out_features=num_reg,
                ),
                nn.Softplus()
            )
        if object_type == 'cls' or object_type == 'reg_cls' or object_type == 'cls_reg':
            self.cls = nn.Sequential(
                nn.Linear(
                    in_features=2 * list_GRU_hidden[-1],
                    out_features=num_cls,
                ),
                nn.Softmax(dim=1)
            )
        if object_type == 'reg_cls' or object_type == 'cls_reg':
            self.mu_cls= nn.Sequential(
                nn.Linear(
                    in_features=num_reg,
                    out_features=num_cls,
                ),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        if self.len_spectrum != self.raw_len:
            total_padding = self.len_spectrum - self.raw_len
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding
            x = F.pad(x, (left_padding, right_padding), mode='replicate')
            # x =F.pad(x, (left_padding, right_padding), mode='constant', value=0)
        B, L = x.size()                            # [batch_size, 3450]
        
        # 通过GRU模块
        # x = self.GRU_list(x.view(B, self.n_subsequence, -1))
        x = x.view(B, self.n_subsequence, -1)      # [batch_size, n_subsequence, 230]
        for gru_block in self.GRU_list:
            x = gru_block(x)            
        
        # 应用注意力机制
        attn_output, _ = self.attention(x,x,x)      # [batch_size, n_subsequence, 64]

        # 求平均
        x = torch.mean(attn_output, dim=1)         # [batch_size, 64]
        x = F.relu(x)
        out_put={}
        if self.object_type == 'reg' or self.object_type == 'reg_cls' or self.object_type == 'cls_reg':

            mu = self.mu(x)
            sigma = self.sigma(x)
            out_put['reg'] = mu
            out_put['sigma'] = sigma
        if self.object_type == 'cls' or self.object_type == 'reg_cls' or self.object_type == 'cls_reg':
            cls = self.cls(x) 
            out_put['cls'] = cls
        if self.object_type == 'reg_cls' or self.object_type == 'cls_reg':
            cls = 0.5*(cls + self.mu_cls(mu))
            out_put['cls'] = cls
        return out_put
    

    



if __name__ == '__main__':
    

    net = BGANet(n_subsequence = 15, 
                 len_spectrum=3446, 
                 list_GRU_hidden=[64, 32], 
                 dropout=0.2,
                 dropout_self=0.,
                 num_label = 3,
                 num_heads = 1)
    x = torch.randn(32, 3446)
    # x = torch.zeros(1,3, 224, 224).cuda()
    out = net(x)
    print(out[0].shape, out[1].shape)
