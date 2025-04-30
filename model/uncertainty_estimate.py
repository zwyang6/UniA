from os import path
from webbrowser import get
import torch 
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

class CTCHead(nn.Module):
    def __init__(self, in_dim, out_dim=4096, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        # pdb.set_trace()
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            # trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class Parameterise(nn.Module):
    """ The unit that unifies information from both low resolution and high resolution branch"""

    def __init__(self,in_dim,hidden_dim,num_heads,pooling_size=4,drop=0.1):
        super().__init__()

        self.in_dim = in_dim
        self.num_heads   = num_heads 
        self.pooling = nn.MaxPool2d((pooling_size,pooling_size),(pooling_size,pooling_size))
        self.mean_cross_attn = nn.MultiheadAttention(embed_dim=in_dim,num_heads=num_heads,batch_first=True)
        self.std_cross_attn = nn.MultiheadAttention(embed_dim=in_dim,num_heads=num_heads,batch_first=True)

        self.mean_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1, groups=in_dim)
        self.std_conv  = nn.Conv2d(in_dim, in_dim, kernel_size=1, groups=in_dim)

        self.std_proj = CTCHead(in_dim=in_dim, out_dim=hidden_dim)
        self.mean_proj = CTCHead(in_dim=in_dim, out_dim=hidden_dim)

    def forward(self,x):
        b,hw,c = x.shape
        x_2d = x.transpose(2,1).reshape(b,c,int(math.sqrt(hw)),int(math.sqrt(hw)))
        mean_c = self.mean_conv(x_2d).reshape(b,c,-1).transpose(2,1)
        std_c  = self.std_conv(x_2d).reshape(b,c,-1).transpose(2,1)

        x_ =  self.pooling(x_2d).reshape(b,c,-1).transpose(2,1) # (B,C,8,8)
        mean_a = self.mean_cross_attn(x,x_,x_)[0]
        std_a = self.std_cross_attn(x,x_,x_)[0] 

        std = std_a+std_c
        mean = mean_a+mean_c

        std = self.std_proj(std)
        mean = self.std_proj(mean)

        return std, mean 


class get_uncertainty(nn.Module):
    def __init__(self,input_dim,num_heads,pooling_size,hidden_dim,BatchNorm=nn.BatchNorm2d,dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_proj = CTCHead(in_dim=input_dim, out_dim=hidden_dim)
        self.conv = nn.Conv2d(hidden_dim,hidden_dim,kernel_size=1)
        self.generate_parameters = Parameterise(in_dim=hidden_dim,hidden_dim=input_dim,num_heads=num_heads,pooling_size=pooling_size)
        self.cross_attn = nn.MultiheadAttention(embed_dim=input_dim,num_heads=num_heads,batch_first=True)

    def reparameterize(self,mean,var,iter=1):
        sample_z = []
        for _ in range(iter):
            std = var.mul(0.5).exp_()  #variable 
            eps = std.data.new(std.size()).normal_()
            sample_z.append(eps.mul(std).add_(mean))
        # sample_z = torch.cat(sample_z, dim=1)
        sample_z = torch.stack(sample_z, dim=1)
        return sample_z
    
    def reparameterize2(self,mean,var,iter=1):
        sample_z = []
        mean_ = mean.unsqueeze(1).repeat(1,iter,1,1)
        var_ = var.unsqueeze(1).repeat(1,iter,1,1)
        std = var_.mul(0.5).exp_()  #variable 
        eps = std.data.new(std.size()).normal_()
        sample_z = (eps.mul(std).add_(mean_))

        return sample_z

    def forward(self,x):
        """
        Input: x [B,C,H,W]
        return: Certainty map and prob_x(estimate map)
        """
        b,hw,c = x.shape
        x_ = self.input_proj(x)
        std, mean = self.generate_parameters(x_)
        uncertainty_map = torch.sigmoid(std)
        residual = x*(1-uncertainty_map)

        x_out = self.cross_attn(x,residual,residual)[0] + x 
        
        prob_x = self.reparameterize(mean,std,50).mean(1)

        return x_out, prob_x


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim 
    model = get_uncertainty(512,512)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.01)
    lr_schudular = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.96)
    x = torch.rand(2,512,224,224)
    label = torch.randn(2,224,224) ** 2
    y,_ = model(x)
    optimizer.zero_grad()
    loss = criterion(y,label.long())
    loss.backward()
    optimizer.step()
    parms = model.parameters()
    print("well-done")