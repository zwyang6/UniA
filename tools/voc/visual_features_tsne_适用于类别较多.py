import argparse
import os
import sys
import logging

sys.path.append("/data/PROJECTS/UniA_2024/TMM_2024/代码/UniA_原始代码/UniA/")

from collections import OrderedDict
import imageio.v2 as imageio
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import voc
from model.model_seg_neg_mct_mask_cle import network
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate, imutils
from sklearn.manifold import TSNE
from matplotlib.colors import BoundaryNorm, ListedColormap



parser = argparse.ArgumentParser()
#! TO DO
## infer valset or testset: The test datafolder is different from valtrain folder
parser.add_argument("--infer_set", default="train", type=str, help="infer_set")
parser.add_argument("--data_folder", default='/data/Datasets/VOC/VOC2012/', type=str, help="dataset folder")
parser.add_argument("--test_data_folder", default='/data/ziqing/Jaye_Files/Dataset/VOC2012/VOCdevkit/VOC2012/', type=str, help="dataset folder")
parser.add_argument("--model_path", default="/data/PROJECTS/UniA_2024/TMM_2024/代码/UniA_原始代码/UniA/00_visual_attn/ours/checkpoints/model_iter_20000.pth", type=str, help="model_path")

parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--scales", default=[1.0,1.25,1.5], help="multi_scales for seg")
parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--decoder", default='largefov', type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling_size", default=4, type=int, help="crop_size in training")
parser.add_argument("--num_samples", default=50, type=int, help="w_reg")

parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

parser.add_argument("--resize_size", default=448, type=int, help="crop_size in training")
parser.add_argument("--aux_layer", default=-3, type=int, help="aux_layer")


def _validate(model=None, data_loader=None, args=None):

    model.eval()
    theme = ['OrRd','RdBu','magma','YlGn']

    color_map = plt.get_cmap(theme[2])

    with torch.no_grad(), torch.cuda.device(0):
        model.cuda()

        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label = data
            inputs  = F.interpolate(inputs, size=[args.resize_size, args.resize_size], mode='bilinear', align_corners=False)

            img = imutils.denormalize_img(inputs)[0].permute(1,2,0).numpy()

            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()
            _, _, h, w = inputs.shape

            H,W = int(h//16),int(w//16)
            labels = F.interpolate(labels.unsqueeze(1).type(torch.float32), size=[H,W], mode="nearest")[0]

            feat = model(inputs)[2]
            feat = feat.flatten(2).permute(0,2,1)
            feat_final = feat[0] # N C
            token_label = labels.reshape(1,-1)[0].cpu().numpy()
            n_cls = np.unique(token_label)

            ####exclude 255
            back = labels.reshape(1,-1)[0].clone().cpu().numpy()
            back[back==255] = -1
            token_label_fore = token_label[back>=0]
            feat_fore = feat_final[back>=0]

            tsne = TSNE(n_components=2,init='pca',random_state=10)
            d = tsne.fit_transform(feat_fore.cpu().numpy())

            num_classes = len(np.unique(token_label_fore))
            colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(np.arange(-0.5, num_classes + 0.5, 1), cmap.N)


            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            ax = plt.gca()
            ax.spines['top'].set_linewidth(2)    # 设置顶部边框粗细
            ax.spines['right'].set_linewidth(2)  # 设置右侧边框粗细
            ax.spines['left'].set_linewidth(2)   # 设置左侧边框粗细
            ax.spines['bottom'].set_linewidth(2) # 设置底部边框粗细
            plt.rcParams['axes.axisbelow'] = True
            plt.grid('on', ls='--',lw=2.4,alpha=0.5)

            axs[0].imshow(img)
            axs[0].axis('off')  # 关闭坐标轴

            scatters = axs[1].scatter(d[:, 0], d[:, 1], c=token_label_fore, s=16, cmap=cmap, norm=norm)
            # 添加颜色条
            cbar = plt.colorbar(scatters, ticks=np.arange(num_classes))
            cbar.ax.set_yticklabels([f"Category {i}" for i in range(num_classes)])
            cbar.set_label("Categories")
            # axs[1].axis('off')  # 关闭坐标轴

            plt.tight_layout()
            plt.savefig(os.path.join(args.tsne_dir, name[0] + ".jpg"),dpi=300)
            plt.close()


            # # 获取当前的Axes对象
            # ax = plt.gca()
            # # 设置每个边框的粗细
            # ax.spines['top'].set_linewidth(2)    # 设置顶部边框粗细
            # ax.spines['right'].set_linewidth(2)  # 设置右侧边框粗细
            # ax.spines['left'].set_linewidth(2)   # 设置左侧边框粗细
            # ax.spines['bottom'].set_linewidth(2) # 设置底部边框粗细
            # plt.rcParams['axes.axisbelow'] = True
            # plt.grid('on', ls='--',lw=2.4,alpha=0.5)
            # plt.scatter(d[:, 0], d[:, 1], c=token_label_fore, s=16)
            # # plt.axis('off')
            # plt.savefig(os.path.join(args.tsne_dir, name[0] + ".jpg"),dpi=300)
            # plt.close()
    
    return


def validate(args=None):

    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.infer_set,
        stage=args.infer_set,
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=False,
                            drop_last=False)

    model = network(args,
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=False,
        aux_layer = -3
    )

    trained_state_dict = torch.load(args.model_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v

    model.load_state_dict(state_dict=new_state_dict, strict=True)
    model.eval()

    _validate(model=model, data_loader=val_loader, args=args)
    torch.cuda.empty_cache()
    
    return True

if __name__ == "__main__":

    args = parser.parse_args()

    base_dir = args.model_path.split("checkpoints/")[0]
    cpt_name = args.model_path.split("checkpoints/")[-1].replace('.pth','')

    args.tsne_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_tsne_img")
    os.makedirs(args.tsne_dir, exist_ok=True)

    print(args)
    validate(args=args)
