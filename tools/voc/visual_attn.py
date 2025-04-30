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


parser = argparse.ArgumentParser()
#! TO DO
## infer valset or testset: The test datafolder is different from valtrain folder
parser.add_argument("--infer_set", default="train", type=str, help="infer_set")
parser.add_argument("--data_folder", default='/data/Datasets/VOC/VOC2012/', type=str, help="dataset folder")
parser.add_argument("--test_data_folder", default='/data/ziqing/Jaye_Files/Dataset/VOC2012/VOCdevkit/VOC2012/', type=str, help="dataset folder")
parser.add_argument("--model_path", default="/data/PROJECTS/UniA_2024/TMM_2024/代码/UniA_原始代码/UniA/00_visual_attn/baseline/checkpoints/baseline_model_iter_20000.pth", type=str, help="model_path")

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

def compute_trans_mat(attn_weight):
    aff_mat = attn_weight

    trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

    for _ in range(2):
        trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
    trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

    for _ in range(1):
        trans_mat = torch.matmul(trans_mat, trans_mat)

    trans_mat = trans_mat

    return trans_mat

def find_foreground_center(mask):

    mask[mask==255] = 0
    mask = mask[0]

    unique_labels, counts = torch.unique(mask, return_counts=True)

    # 去掉背景（标签 0）
    foreground_labels = unique_labels[unique_labels != 0]
    foreground_counts = counts[unique_labels != 0]

    # 如果没有前景，返回None
    if foreground_counts.size(0) == 0:
        return tuple([10,10])

    # 找到拥有最多像素的前景标签
    max_label = foreground_labels[torch.argmax(foreground_counts)]

    # 找到该标签对应的所有坐标
    foreground_coords = torch.nonzero(mask == max_label)

    # 计算这些坐标的均值，得到重心
    center = foreground_coords.float().mean(dim=0)

    center = center.round().int()

    # 检查质心是否在前景内
    if mask[center[0], center[1]] == max_label:
        center = center  # 选择质心
    else:
        # 随机选择一个前景像素作为中心
        center = foreground_coords[torch.randint(0, foreground_coords.size(0), (1,))][0]
    
    # 返回重心的坐标（可以选择四舍五入取整，或保持浮点数形式）
    return center.tolist()

def save_attm_maps(inputs, attn, size, name, save_dir, color_map, anchor=None, anchor_dir=None):
    sal_g = attn.unsqueeze(0).unsqueeze(0)
    # pdb.set_trace()
    sal_g = F.interpolate(sal_g, size=size, mode="bilinear", align_corners=False)[0,0,...].cpu().numpy()
    sal_g -= sal_g.min()
    sal_g /= sal_g.max()
    sal_g += 0.1
    sal_g[sal_g>=1]=1
    # sal_g = sal_g ** (0.5)
    # sal_g += 0.2
    # sal_g[sal_g<=0.65]=sal_g[sal_g<=0.65] / 2
    # sal_g = color_map(sal_g)[:,:,:3] * 255
    # stack_img = np.hstack((inputs, sal_g))
    # imageio.imsave(os.path.join(save_dir, name + ".jpg"), stack_img.astype(np.uint8))

    idx = int(anchor[0]//16) * int(anchor[1]//16)
    idx_row = attn[idx,...]
    anchor_feat = idx_row.reshape(size[0]//16,size[1]//16).unsqueeze(0).unsqueeze(0)
    anchor_feat_resize = F.interpolate(anchor_feat, size=size, mode="bilinear", align_corners=False)[0,0,...].cpu().numpy()
    anchor_feat_resize -= anchor_feat_resize.min()
    anchor_feat_resize /= anchor_feat_resize.max()

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(inputs)
    axs[0].scatter(anchor[1], anchor[0], color='orange', marker='*', s=200)  # 注意x, y的顺序
    axs[0].axis('off')  # 关闭坐标轴

    axs[1].imshow(anchor_feat_resize,cmap='viridis_r')
    axs[1].axis('off')  # 关闭坐标轴

    axs[2].imshow(anchor_feat_resize,cmap='viridis')
    axs[2].axis('off')  # 关闭坐标轴

    axs[3].imshow(sal_g, cmap='YlGn')
    axs[3].axis('off')  # 关闭坐标轴

    plt.tight_layout()
    plt.savefig(os.path.join(anchor_dir, name + ".jpg"),dpi=200)
    plt.close()

    return 

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


            feat = model(inputs)[2]
            feat = feat.flatten(2).permute(0,2,1)
            attn_final = torch.matmul(feat,feat.permute(0,2,1)).detach()[0]
            _trans_mat = compute_trans_mat(attn_final)
            refined_attn_maps = _trans_mat.float()
            labels = F.interpolate(labels.unsqueeze(1).type(torch.float32), size=[args.resize_size, args.resize_size], mode="nearest")[0]
            anchor_coor = find_foreground_center(labels)

            save_attm_maps(inputs=img, attn=refined_attn_maps, size=inputs.shape[2:], name=name[0], \
                            save_dir=args.attn_dir, color_map=color_map, anchor=anchor_coor, anchor_dir=args.anchor_dir)
    
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

    args.attn_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_attn_img")
    args.anchor_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_anchor_img")

    os.makedirs(args.attn_dir, exist_ok=True)
    os.makedirs(args.anchor_dir, exist_ok=True)

    print(args)
    validate(args=args)
