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
        return x

class embedding(nn.Module):
    def __init__(self,
                
                input_channel=1,
                embedding_c=50,
                kernel_size=3,
                overlap=1,
                padding=1,

                 ):
        self.input_channel=input_channel
        self.embedding_c=embedding_c
        self.kernel_size=kernel_size
        self.overlap=overlap
        self.padding=padding
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
        insize=x.size(2)
        self.output_length=self.caculate_output_length(insize)
        x=self.embedding(x)
        x=x.permute(0,2,1)
        # print((insize+2*self.padding-self.kernel_size)//(self.kernel_size-self.overlap)+1,'计算的长度')
        return x
    
    def caculate_output_length(self, input_length):
        """
        计算输出长度
        """
        output_length = (input_length + 2 * self.padding - self.kernel_size) // (self.kernel_size - self.overlap) + 1
        return output_length

class MPBDNet(nn.Module):
    def __init__(self,
                num_cls=3,#分类的种类数
                num_reg=5,#回归的变量数
                list_inplanes= [3,6,18],
                num_rnn_sequence = 18,
                embedding_c=50,
                len_spectrum=4900,
                object_type="cls",
                blocks_length=3
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
        
        for i in range(len(self.list_inplanes) - 1):
            block_sequence = []
            for _ in range(blocks_length):
                block_sequence.append(MPBDBlock(self.list_inplanes[i], self.list_inplanes[i + 1]))
                self.list_inplanes[i] = self.list_inplanes[i + 1]
            block_sequence.append(nn.AvgPool1d(3))
            self.MPBDBlock_list.append(nn.Sequential(*block_sequence))

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
        self.fc1_input_length = self.caculate_fc_input_length(len_spectrum)
        # print(f"FC1输入维度: {self.fc1_input_length}")
        self.fc1 = nn.Sequential(
            nn.Linear(
                in_features=self.fc1_input_length,
                out_features=256,
            ),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        if self.object_type=="cls" or self.object_type=="reg_cls" or self.object_type=="cls_reg":
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
                out_features=num_cls,
            )
        if self.object_type=="reg" or self.object_type=="reg_cls" or self.object_type=="cls_reg":
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
                out_features=num_reg,
            )

    def caculate_fc_input_length(self, input_length):
        """
        计算 fc1 的输入维度
        """
        # 模拟前向传播过程计算输出长度
        x = torch.randn(1, 1, input_length)
        x = self.MPBDBlock_list(x)
        x = torch.relu(x)
        
        x = self.embeding(x)
        x = (self.rnn(x)[0] + torch.flip(self.rnn2(torch.flip(x, dims=[1]))[0], dims=[1])) / 2
        x = x.flatten(1)
        return x.size(1)
    
    def caculate_fc_input_length_fast(self, input_length):
        """
        计算 fc1 的输入维度
        """
        # 手动计算每一层的输出尺寸
        for block_seq in self.MPBDBlock_list:
            for layer in block_seq:
                if isinstance(layer, nn.Conv1d):
                    input_length = (input_length + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
                elif isinstance(layer, nn.AvgPool1d):
                    input_length = (input_length + 2 * layer.padding - layer.kernel_size) // layer.stride + 1

        # 计算 embedding 层的输出长度
        input_length = self.embeding.caculate_output_length(input_length)

        # LSTM 层输出尺寸不变
        # 展平后的输入维度
        fc_input_length = input_length * self.embedding_c
        return fc_input_length


    def forward(self, x):
        # print(x.shape)
        if x.size(1)%8!=0:
            x=torch.cat([x, torch.zeros([x.size(0),x.size(1)%8]).to(x.device)], dim=1)
        x = x.unsqueeze(1)
        x = self.MPBDBlock_list(x)
        x = torch.relu(x)
        x = self.embeding(x)
        x=(self.rnn(x)[0]+torch.flip(self.rnn2(torch.flip(x,dims=[1]))[0],dims=[1]))/2
        # print(f"展平后的张量形状: {x.flatten(1).shape}")
        x_flattened = self.fc1(x.flatten(1))
        out_put={}
        if self.object_type == "cls":
            cls_output = self.cls(self.cls_benck(x_flattened))
            out_put['cls'] = cls_output
        elif self.object_type == "reg":
            reg_output = self.reg(self.reg_benck(x_flattened))
            out_put['reg'] = reg_output
        elif self.object_type == "reg_cls" or self.object_type == "cls_reg":
            cls_output = self.cls(self.cls_benck(x_flattened))
            reg_output = self.reg(self.reg_benck(x_flattened))
            out_put['cls'] = cls_output
            out_put['reg'] = reg_output

        return out_put

if __name__ == "__main__":
    # 创建模型实例
    model = MPBDNet(
        num_classes=3,
        list_inplanes=[3, 6, 18],
        len_spectrum=4900,
        object_type="cls"
    )
    
    # 创建随机输入数据
    x = torch.randn(2, 3834)  # 批次大小为2，序列长度为3834
    
    # 前向传播
    print("模型测试中...")
    output = model(x)
    
    # 打印输出结果
    if isinstance(output, tuple):
        print(f"分类输出形状: {output[0].shape}")
        print(f"回归输出形状: {output[1].shape}")
    else:
        print(f"输出形状: {output.shape}")
    print("测试完成!")    