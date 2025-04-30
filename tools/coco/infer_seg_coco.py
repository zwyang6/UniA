import argparse
import os
import sys
import logging

sys.path.append(".")

from collections import OrderedDict
import imageio.v2 as imageio
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import coco2 as coco
from model.model_seg_neg_mct_mask_cle import network
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate, imutils
from utils.dcrf import DenseCRF
from utils.pyutils import format_tabs_multi_metircs, setup_logger

parser = argparse.ArgumentParser()
#! TO DO
## infer valset or testset: The test datafolder is different from valtrain folder
parser.add_argument("--infer_set", default="val", type=str, help="infer_set")
parser.add_argument("--data_folder", default='/home/jaye/Documents/Datasets/MSCOCO/', type=str, help="dataset folder")
parser.add_argument("--model_path", default="/data/ziqing/Jaye_Files/SeCo_cvpr24_rebuttal/w_outputs/2024-01/prototype_only_bkg_and_fore_23-33-18/checkpoints/model_iter_20000.pth", type=str, help="model_path")

parser.add_argument("--list_folder", default='datasets/coco', type=str, help="train/val/test list file")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--scales", default=[1.0,1.25,1.5], help="multi_scales for seg")
parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--decoder", default='largefov', type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling_size", default=4, type=int, help="crop_size in training")
parser.add_argument("--num_samples", default=50, type=int, help="w_reg")

parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")
parser.add_argument("--num_classes", default=81, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")
parser.add_argument("--aux_layer", default=-3, type=int, help="aux_layer")

def colorful(out,name):
    arr=out.astype(np.uint8)
    im=Image.fromarray(arr)
 
    palette=[]
    for i in range(256):
        palette.extend((i,i,i))
    palette[:3*21]=np.array([[0, 0, 0],
                                [128, 0, 0],
                                [0, 128, 0],
                                [128, 128, 0],
                                [0, 0, 128],
                                [128, 0, 128],
                                [0, 128, 128],
                                [128, 128, 128],
                                [64, 0, 0],
                                [192, 0, 0],
                                [64, 128, 0],
                                [192, 128, 0],
                                [64, 0, 128],
                                [192, 0, 128],
                                [64, 128, 128],
                                [192, 128, 128],
                                [0, 64, 0],
                                [128, 64, 0],
                                [0, 192, 0],
                                [128, 192, 0],
                                [0, 64, 128]
                             ], dtype='uint8').flatten()
 
    im.putpalette(palette)
    im.save(name)


def _validate(model=None, data_loader=None, args=None):

    model.eval()
    color_map = plt.get_cmap("Blues")

    with torch.no_grad(), torch.cuda.device(0):
        model.cuda()

        gts, seg_pred = [], []

        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()
            _, _, h, w = inputs.shape
            seg_list = []
            for sc in args.scales:
                _h, _w = int(448*sc), int(448*sc)

                _inputs  = F.interpolate(inputs, size=[_h, _w], mode='bilinear', align_corners=False)
                # inputs_cat = _inputs
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)
                
                segs = model(inputs_cat,)[1]
                segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

                # seg = torch.max(segs[:1,...], segs[1:,...].flip(-1))
                seg = segs[:1,...] + segs[1:,...].flip(-1)
                # seg = segs

                seg_list.append(seg)
            seg = torch.max(torch.stack(seg_list, dim=0), dim=0)[0]
            seg_pred += list(torch.argmax(seg, dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

            np.save(args.logits_dir + "/" + name[0] + '.npy', {"msc_seg":seg.cpu().numpy()})

    seg_score = evaluate.scores(gts, seg_pred, num_classes=81)
    logging.info('raw_seg_score:')
    metrics_tab = format_tabs_multi_metircs([seg_score], ["confusion","precision","recall",'iou'], cat_list=coco.class_list)
    logging.info("\n"+metrics_tab)

    
    return seg_score


def crf_proc():
    print("crf post-processing...")

    txt_name = os.path.join(args.list_folder, args.infer_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    if "val" in args.infer_set:
        images_path = f"{args.data_folder}/JPEGImages/val/"
        labels_path = f"{args.data_folder}/SegmentationClass/val/"
    elif "train" in args.infer_set:
        images_path = f"{args.data_folder}/JPEGImages/train/"
        labels_path = f"{args.data_folder}/SegmentationClass/train/"

    post_processor = DenseCRF(
        iter_max=10,    # 10
        pos_xy_std=1,   # 3
        pos_w=1,        # 3
        bi_xy_std=121,  # 121, 140
        bi_rgb_std=5,   # 5, 5
        bi_w=4,         # 4, 5
    )

    def _job(i):

        name = name_list[i]

        logit_name = args.logits_dir + "/" + name + ".npy"

        logit = np.load(logit_name, allow_pickle=True).item()
        logit = logit['msc_seg']

        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)
        if len(image.shape)<3:
            image = np.stack((image, image, image), axis=-1)
        label_name = os.path.join(labels_path, name[13:] + ".png")
        if "test" in args.infer_set:
            label = image[:,:,0]
        else:
            # label = imageio.imread(label_name)
            label = np.asarray(Image.open(label_name))

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)#[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)

        imageio.imsave(args.segs_rgb_dir + "/" + name + ".png", imutils.encode_cmap(np.squeeze(pred)).astype(np.uint8))
        
        if args.infer_set == 'test':
            colorful(np.squeeze(pred).astype(np.uint8),args.test_segs_dir + "/" + name + ".png")
        return pred, label
    
    n_jobs = int(os.cpu_count() * 0.6)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])

    preds, gts = zip(*results)

    crf_score = evaluate.scores(gts, preds, num_classes=81)
    logging.info('crf_seg_score:')
    metrics_tab_crf = format_tabs_multi_metircs([crf_score], ["confusion","precision","recall",'iou'], cat_list=coco.class_list)
    logging.info("\n"+ metrics_tab_crf)

    return crf_score

def validate(args=None):

    val_dataset = coco.COCOSegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.infer_set,
        stage='val',
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

    trained_state_dict = torch.load(args.model_path, map_location="cpu")['model_state_dict']

    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        k = k.replace('module.', '')
        new_state_dict[k] = v

    model.load_state_dict(state_dict=new_state_dict, strict=True)
    model.eval()

    # seg_score = _validate(model=model, data_loader=val_loader, args=args)
    torch.cuda.empty_cache()

    crf_score = crf_proc()
    
    return True

if __name__ == "__main__":

    args = parser.parse_args()

    base_dir = args.model_path.split("checkpoints/")[0]
    cpt_name = args.model_path.split("checkpoints/")[-1].replace('.pth','')
    args.logits_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_segs/logits")
    args.segs_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_segs/seg_preds")
    args.segs_rgb_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_segs/seg_preds_rgb")

    os.makedirs(args.segs_dir, exist_ok=True)
    os.makedirs(args.segs_rgb_dir, exist_ok=True)
    os.makedirs(args.logits_dir, exist_ok=True)

    ### for test 
    if args.infer_set == 'test':
        args.test_segs_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_segs/website_test_form/results/VOC2012/Segmentation/comp6_test_cls/")
        os.makedirs(args.test_segs_dir, exist_ok=True)
        args.data_folder = args.test_data_folder

    setup_logger(filename=os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_segs/results.log"))

    print(args)
    validate(args=args)