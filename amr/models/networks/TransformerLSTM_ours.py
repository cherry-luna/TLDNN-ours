from torch.autograd import Variable
import torch
from torchsummaryX import summary
import math
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from fvcore.nn import FlopCountAnalysis, flop_count_table
from thop import profile
from thop import clever_format
from einops import rearrange, repeat
import torch.nn.functional as F
__all__ = ["TransformerLSTM"]


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y

class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class FeedForward_GLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.ReLU()
        self.fc3 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x = self.gelu(x1)*x2
        f = self.fc3(x)
        return f

class FeedForward_2headsGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc3 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        f1 = x1*self.gelu(x2)
        f2 = x2*self.gelu(x1)
        f = self.fc3(f1+f2)
        return f

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=8, dropout=0., talk_heads=True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.talking_heads1 = nn.Conv2d(heads, heads, 1, bias=False) if talk_heads else nn.Identity()
        self.talking_heads2 = nn.Conv2d(heads, heads, 1, bias=False) if talk_heads else nn.Identity()

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = self.talking_heads1(dots)

        attn = self.attend(dots)
        attn = self.dropout(attn)
        attn = self.talking_heads2(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, talk_heads, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, talk_heads=talk_heads),
                #nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=False),
                nn.LayerNorm(dim),
                FeedForward_GLU(dim, mlp_dim, dropout=dropout),
                nn.LayerNorm(dim)
            ]))

    def forward(self, x):
        for attn, norm1, ff, norm2 in self.layers:
            x = norm1(attn(x) + x)
            #x = norm1(attn(x,x,x,need_weights=False, average_attn_weights=False)[0] + x)
            x = norm2(ff(x) + x)
        return x

class GroupBlock(nn.Module):
    def __init__(self, in_channel, hidden_channel):
        super(GroupBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, hidden_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channel, hidden_channel, kernel_size=5, padding=2, groups=in_channel),
            nn.ReLU(),
            nn.Conv1d(hidden_channel, in_channel, kernel_size=3, padding=1),
        )
        self.shortcut = nn.Sequential(
            nn.Identity(),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.shortcut(x)
        x3 = self.relu(x1 + x2)
        return x3

    def __call__(self, x):
        return self.forward(x)

class Denoise(nn.Module):
    def __init__(self, in_channel, hidden_channel,p=1):
        super(Denoise, self).__init__()
        self.p = p
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.L1 = nn.Linear(in_channel, hidden_channel)
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(hidden_channel)
        self.L2 = nn.Linear(hidden_channel, in_channel)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x.shape=(B,in_channel,2,L)
        y = self.global_avg_pool(x) # y.shape=(B,in_channel,1,1)
        y = y.view(y.shape[0],y.shape[1]) # 将 y 的形状从 (B, in_channel, 1, 1) 调整为 (B, in_channel)
        y = self.L1(y)
        y = self.relu(y)
        y = y.view(y.shape[0],y.shape[1],1,1) # 将 y 的形状从 (B, hidden_channel) 调整为 (B, hidden_channel, 1, 1)
        y = self.BN(y)
        y = y.view(y.shape[0],y.shape[1]) # 将 y 的形状从 (B, hidden_channel, 1, 1) 调整为 (B, hidden_channel)
        y = self.L2(y)
        a = self.sigmoid(y) # a.shape=(B,in_channel)
        beta = a*y
        threshold = beta*(1+self.p)
        new_threshold = threshold.view(threshold.shape[0], threshold.shape[1], 1, 1)
        x_abs = torch.abs(x)
        # 创建掩码，形状与 x 相同, 将 threshold 的形状从 (B,in_channel) 调整为 (B, in_channel, 1, 1)
        mask = x_abs >= new_threshold
        # 如果 mask 为 True，则更新为 sgn(x) * (x_abs - new_threshold)。如果 mask 为 False，则更新为 0。
        x = torch.where(mask, torch.sign(x) * (x_abs - new_threshold), torch.tensor(0.0, dtype=x.dtype, device=x.device))
        return x
class TADBlock(nn.Module):
    def __init__(self, c1=64, c2=16):
        super(TADBlock, self).__init__()
        self.BN = nn.BatchNorm2d(1)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=(2,3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=(2,3), padding='same'))
        self.decoder = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=(2,3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(c2, c1, kernel_size=(2,3), padding='same'),
            nn.Conv2d(c1, 1, kernel_size=(2,3), padding='same'))
        self.denoise = Denoise(in_channel=c2, hidden_channel=c2*4,p=1)
    def forward(self, x):
        if x.ndim == 3 and x.shape[1] == 2:  # 检查形状是否为 B*2*L
            x = x.unsqueeze(1)  # 在维度 1 处添加一个维度，变为 B*1*2*L
        x = self.BN(x)
        x = self.encoder(x)
        x = self.denoise(x)
        x = self.decoder(x)
        if x.ndim == 4 and x.shape[1] == 1:  # 检查形状是否为 B*1*2*L
            x = x.squeeze(1)  # 移除维度 1
        return x
        

class TransformerLSTM(nn.Module):
    def __init__(self, classes: int, d_model: int = 64, nhead: int = 8, d_hid: int = 128, lstm_hidden_size: int = 64,
                 nlayers: int = 2 , dropout: float = 0.2, max_len: int = 5000, mask=False, poolsize = (2,1)):
        super().__init__()
        self.model_type = 'TransformerLSTM'
        self.mask = mask
        self.d_model = d_model
        self.TADBlock = TADBlock()
        self.Convlayer = nn.Sequential(
                    nn.Conv1d(2, d_model, kernel_size = 4, stride = 2, padding = 1),
                    nn.ReLU(),
                    nn.Conv1d(d_model, d_model, kernel_size = 4, stride = 2, padding = 1),
                    nn.ReLU(),
                    nn.Conv1d(d_model, d_model, kernel_size = 4, stride = 2, padding = 1),
                    nn.ReLU(),
                    nn.Conv1d(d_model, d_model, kernel_size = 4, stride = 2, padding = 1),
                    )
        self.pos_embedding = nn.Parameter(torch.randn(1, 256, d_model))  #! 其中，256=4096/(2^卷积层层数)
        self.se = SEAttention(channel=d_model, reduction=4)

        self.transformer_encoder =  nn.Sequential(
                    Transformer(dim=d_model, depth=nlayers, heads=nhead, dim_head=d_model//nhead, mlp_dim=d_hid, talk_heads=True, dropout=dropout),
                    )

        self.lstm_layer = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden_size, num_layers=4, dropout=dropout, batch_first=True)        
        self.decoder = nn.Sequential(
            nn.Linear(in_features=lstm_hidden_size, out_features=lstm_hidden_size),
            nn.ReLU(),  # 激活函数
            nn.BatchNorm1d(lstm_hidden_size),  # 批量归一化,
            nn.Dropout(dropout),
            
            nn.Linear(in_features=lstm_hidden_size, out_features=int(lstm_hidden_size/2)),
            nn.ReLU(),  # 激活函数
            nn.BatchNorm1d(int(lstm_hidden_size/2)),  # 批量归一化
            nn.Dropout(dropout),

            nn.Linear(in_features=int(lstm_hidden_size/2), out_features=classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float() #! x.shape = [B, 2, L]
        x = self.TADBlock(x)
        x = self.Convlayer(x) #! x.shape = [B, d_model, N], N=L/(2^卷积层层数)
        N = x.shape[2] #! 
        
        x = self.se(x)

        x += self.pos_embedding[:, :N, :].transpose(1,2)

        #! x.shape=[B, d_model，N]
        x = self.transformer_encoder(x.transpose(1,2))
        dec_out, (hidden_state, cell_state) = self.lstm_layer(x)
        fc = self.decoder(hidden_state[-1])

        return fc

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

if __name__ == '__main__':
    net = TransformerLSTM(11)
    batchsize = 512
    data_input = Variable(torch.randn([batchsize, 2, 1024]))
    #summary(net, data_input)
    #net(data_input)
    net.eval()
    print(flop_count_table(FlopCountAnalysis(net, data_input)))
    flops, params = profile(net, inputs=(data_input, ))
    flops,params = clever_format([flops, params],"%.3f")
    print(params,flops) 
