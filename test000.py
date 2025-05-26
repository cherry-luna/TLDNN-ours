import torch
import os        
pretrained = '/TLDNN-ours/results/TransformerLSTM/best/ours/checkpoints/dyn_pad_best_acc.pth'
assert os.path.isfile(pretrained)
pre_state_dict = torch.load(pretrained, map_location=torch.device('cpu'))['state_dict']
# 只加载和当前模型结构一致的参数
for name, param in pre_state_dict.items():
    print(name,":",param.shape)