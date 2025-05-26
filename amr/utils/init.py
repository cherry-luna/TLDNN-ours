import torch
import numpy as np
import random
import os
from amr.utils import logger
from amr.models import *
import importlib

__all__ = ["init_device", "init_model", "init_loss"]


def init_device(seed=None, cpu=None, gpu=None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(seed)
        pin_memory = True
        logger.info("Running on GPU%d" % (gpu if gpu else 0))
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info("Running on CPU")

    return device, pin_memory


def init_model(args):
    if args.modes.train:
        model = getattr(
                importlib.import_module("amr.models.networks." + args.modes.method + '_' + args.data_settings.dataset),
                args.modes.method)(len(args.data_settings.mod_type)) # 11类
        state_dict = model.state_dict() 
        if args.modes.load_pretrained:
            pretrained = args.modes.load_pretrained_path
            assert os.path.isfile(pretrained)
            pre_state_dict = torch.load(pretrained, map_location=torch.device('cpu'))['state_dict']
            # 只加载和当前模型结构一致的参数
            for name, param in pre_state_dict.items():
                if name in state_dict and state_dict[name].shape == param.shape:
                    state_dict[name] =param           
            model.load_state_dict(state_dict,strict=False)
            print("pretrained model loaded from {}".format(pretrained))
        if args.modes.load_pretrained_pos:
            # 加载预训练模型pos_embedding的部分参数，并线性插值到1*256*n_model的形状
            pretrained = './results/TransformerLSTM/best/RML2018/checkpoints/best_acc.pth'
            assert os.path.isfile(pretrained)
            pre_pos_state_dict = torch.load(pretrained, map_location=torch.device('cpu'))['state_dict']
            pre_pos_embedding = pre_pos_state_dict['pos_embedding']
            pre_pos_embedding = pre_pos_embedding.permute(0, 2, 1)  # [1, n_model, 64]
            new_length = state_dict['pos_embedding'].shape[1]
            interpolated = torch.nn.functional.interpolate(
                pre_pos_embedding,
                size=new_length,
                mode='linear',  # 线性插值
                align_corners=False  # 是否对齐角点
            )
            # 恢复原始形状 [1, new_length, n_model]
            interpolated = interpolated.permute(0, 2, 1)
            state_dict['pos_embedding'] = interpolated 
            model.load_state_dict(state_dict,strict=False)
            print("pretrained pos_embedding loaded from {}".format(pretrained))

        if args.modes.load_pretrained_TAD:
            pretrained = './results/TransformerLSTM/best/ours/11types/checkpoints/best_acc_15.pth'
            assert os.path.isfile(pretrained)
            # 加载整个预训练模型的state_dict
            pretrained_state_dict = torch.load(pretrained, map_location=torch.device('cpu'))
            # for k in state_dict.keys():
            #     if 'TADBlock' in k:
            #         print(k)
            # 提取属于TADBlock部分的参数
            tad_params = {
                k[len('TADBlock.'):]: v  # 去除前缀'TADBlock.'
                for k, v in pretrained_state_dict.items() 
                if k.startswith('TADBlock.')
            }
            # 加载到当前模型的TADBlock中
            model.TADBlock.load_state_dict(tad_params, strict=False)
            print("pretrained TADBlock loaded from {}".format(pretrained))
        print(model)
        
    if not args.modes.train:
        model = getattr(
                importlib.import_module("amr.models.networks." + args.modes.method + '_' + args.data_settings.dataset),
                args.modes.method)(len(args.data_settings.mod_type)) # 11类
        pretrained = 'results/' + args.modes.method + '/' + args.modes.path + '/' + args.data_settings.dataset + '/11types/checkpoints/best_acc.pth'
        assert os.path.isfile(pretrained)
        state_dict = torch.load(pretrained, map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(state_dict)
        logger.info("pretrained model loaded from {}".format(pretrained))

    return model


def init_loss(loss_func):
    loss = getattr(importlib.import_module("amr.models.losses.loss"), loss_func)()
    return loss


if __name__ == '__main__':
    model = getattr(importlib.import_module("amr.models.networks." + 'ResNet_RML2016'), "ResNet")(11)
    print(model)
