import argparse
import datetime
import logging
import os
import random
import sys

sys.path.append(".")
from datasets.transforms import MultiviewTransform
import numpy as np
import torch
import torch.nn as nn 
import torch.distributed as dist
import torch.nn.functional as F
from datasets import coco
from model.losses import get_masked_ptc_loss, get_seg_loss, DenseEnergyLoss, get_energy_loss, get_aff_loss, KLloss
from model.model_seg_neg_mct_mask_cle import network
from model.dense_refinement import dr
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from model.PAR import PAR
from utils import evaluate, imutils, optimizer
from utils.camutils import cam_to_label,multi_scale_cam2, label_to_aff_mask, refine_cams_with_bkg_v2,get_mask_by_radius, cams_to_affinity_label
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger, resume_train
torch.hub.set_dir("/data/ziqing/Jaye_Files/00_RoCo_Rebuttal/pretrained")
parser = argparse.ArgumentParser()
from torch.utils.tensorboard import SummaryWriter
from utils.tbutils import make_grid_image, make_grid_label
from utils.affutils import multi_scale_cam_with_aff,propagte_aff_cam_with_bkg,refine_cam_with_aff,refine_cams_with_bkg_aff

parser.add_argument("--use_aa", type=bool, default=False)
parser.add_argument("--use_gauss", type=bool, default=False)
parser.add_argument("--use_solar", type=bool, default=False)

parser.add_argument("--work_dir", default="w_outputs", type=str, help="work_dir_voc_wseg")
parser.add_argument("--log_tag", default="sota_mct_72.5", type=str, help="work_dir_voc_wseg")
parser.add_argument("--save_ckpt", default=True, action="store_true", help="save_ckpt")
parser.add_argument("--tensorboard", default=False, type=bool, help="log tb")
parser.add_argument("--resume_train", default=True, type=lambda x: x.lower() in ["true", "1", "yes"], help="log tb")
parser.add_argument("--resume_path", default="/HOME/scz0658/run/Jaye_Files/unicam/unicam/w_outputs/coco/2024-03/coco_train_seed7_31-20-25-58/checkpoints/model_iter_latest.pth", type=str, help="log tb")

parser.add_argument("--w_ptc", default=0.3, type=float, help="w_ptc")
parser.add_argument("--w_kl", default=0.1, type=float, help="w_ctc")
parser.add_argument("--w_aff", default=0.1, type=float, help="w_ptc")
parser.add_argument("--w_seg", default=0.12, type=float, help="w_seg")
parser.add_argument("--w_reg", default=0.05, type=float, help="w_reg")
parser.add_argument("--num_samples", default=50, type=int, help="w_reg")

parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--decoder", default='largefov', type=str, help="vit_base_patch16_224")     
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")

parser.add_argument("--data_folder", default='/data/ziqing/Jaye_Files/Dataset/MSCOCO/', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/coco', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=81, type=int, help="number of classes")
parser.add_argument("--radius", default=7, type=int, help="crop_size in training")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")
parser.add_argument("--pooling_size", default=4, type=int, help="crop_size in training")

parser.add_argument("--train_set", default="train", type=str, help="training split")
parser.add_argument("--val_set", default="val_part", type=str, help="validation split")
parser.add_argument("--spg", default=4, type=int, help="samples_per_gpu")
parser.add_argument("--scales", default=(0.5, 2), help="random rescale in training")

parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=6e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")

parser.add_argument("--max_iters", default=80000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")
parser.add_argument("--aux2final", default=30000, type=int, help="use dense refined masks to refine aff")
parser.add_argument("--seg2final", default=30000, type=int, help="use dense refined masks to refine aff")
parser.add_argument("--begin_val", default=60000, type=int, help="use dense refined masks to refine aff")

parser.add_argument("--high_thre", default=0.6, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.45, type=float, help="bkg_score")
parser.add_argument("--cam_scales", default=[1.0,0.5,0.75,1.25,1.5], help="multi_scales for cam")

parser.add_argument("--temp", default=0.5, type=float, help="temp")
parser.add_argument("--momentum", default=0.9, type=float, help="temp")
parser.add_argument("--aux_layer", default=-3, type=int, help="aux_layer")

parser.add_argument("--seed", default=0, type=int, help="fix random seed")

parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--num_workers", default=10, type=int, help="num_workers")
parser.add_argument('--backend', default='nccl')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def validate(model=None, data_loader=None, args=None):

    preds, gts, cams, cams_aux, cams_aff, cams_dr = [], [], [], [],[], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()


            inputs  = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)

            cls, segs, _, _, attn_pred = model(inputs)

            cls_pred = (cls>0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            _cams, _cams_aux = multi_scale_cam2(model, inputs, args.cam_scales)
            # _cams, _cams_aux, attns = multi_scale_cam_with_aff(model, inputs, args.cam_scales, cls_labels=None)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            valid_cam, cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            ## aff_label
            valid_cam_resize = F.interpolate(valid_cam, size=segs.shape[-2:], mode='bilinear', align_corners=False) 
            attn_mask = get_mask_by_radius(h=segs.shape[-2],w=segs.shape[-1],radius=args.radius)
            aff_cam = propagte_aff_cam_with_bkg(valid_cam_resize, aff=attn_pred, mask=attn_mask, cls_labels=cls_label, bkg_score=0.5)
            aff_cam = F.interpolate(aff_cam, size=labels.shape[1:], mode="bilinear", align_corners=False)
            aff_label = aff_cam.argmax(dim=1)

            dr_label = dr(20000,cam_label,cam_label,aff_label)

            resized_cam_aux = F.interpolate(_cams_aux, size=labels.shape[1:], mode='bilinear', align_corners=False)
            _, cam_label_aux = cam_to_label(resized_cam_aux, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            cams_aux += list(cam_label_aux.cpu().numpy().astype(np.int16))
            cams_aff += list(aff_label.cpu().numpy().astype(np.int16))
            cams_dr += list(dr_label.cpu().numpy().astype(np.int16))

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds, num_classes=args.num_classes)
    cam_score = evaluate.scores(gts, cams, num_classes=args.num_classes)
    cam_aux_score = evaluate.scores(gts, cams_aux, num_classes=args.num_classes)
    cam_aff_score = evaluate.scores(gts, cams_aff, num_classes=args.num_classes)
    cam_dr_score = evaluate.scores(gts, cams_dr, num_classes=args.num_classes)
    model.train()

    tab_results = format_tabs([cam_score, cam_aux_score, cam_aff_score, cam_dr_score, seg_score], name_list=["CAM", "aux_CAM", "aff_Map", "dr_Map", "Seg_Pred"], cat_list=coco.class_list)

    return cls_score, tab_results

def train(args=None):

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, )
    logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))

    if args.local_rank == 0 and args.tensorboard == True:
        tb_logger = SummaryWriter(log_dir=args.tb_dir)

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_transform = MultiviewTransform(
            size1=args.crop_size,
            size2=args.crop_size,
            num1=2,
            num2=1,
            use_aa=args.use_aa,
            use_gauss=args.use_gauss,
            use_solar=args.use_solar,
        )
    
    train_dataset = coco.COCOClsDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split="train",
        stage="train",
        ignore_index=args.ignore_index,
        transform=train_transform,
        aug=True
    )

    val_dataset = coco.COCOSegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.val_set,
        stage='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        #shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=False)

    device = torch.device(args.local_rank)

    model = network(args,
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        init_momentum=args.momentum,
        aux_layer=args.aux_layer
    )
    param_groups = model.get_param_groups()

    begin_iter = 0
    if os.path.exists(args.resume_path) and args.resume_train:
        model, optim, begin_iter = resume_train(args,model,optimizer)
        if args.local_rank == 0:
            logging.info(f'Resuming_{begin_iter}iter_Training_from_{args.resume_path}...')
    else:
        optim = getattr(optimizer, args.optimizer)(
            params=[
                {
                    "params": param_groups[0],
                    "lr": args.lr,
                    "weight_decay": args.wt_decay,
                },
                {
                    "params": param_groups[1],
                    "lr": args.lr,
                    "weight_decay": args.wt_decay,
                },
                {
                    "params": param_groups[2],
                    "lr": args.lr * 10,
                    "weight_decay": args.wt_decay,
                },
                {
                    "params": param_groups[3],
                    "lr": args.lr * 10,
                    "weight_decay": args.wt_decay,
                },
            ],
            lr=args.lr,
            weight_decay=args.wt_decay,
            betas=args.betas,
            warmup_iter=args.warmup_iters,
            max_iter=args.max_iters,
            warmup_ratio=args.warmup_lr,
            power=args.power)

    model.to(device)
    logging.info('\nOptimizer: \n%s' % optim)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    train_sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()


    loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)

    par = PAR(num_iter=10, dilations=[1,2,4,8,12,24]).cuda()

    for n_iter in range(begin_iter, args.max_iters):
        global_step = n_iter + 1

        try:
            img_name, inputs, cls_label, img_box, crops = next(train_loader_iter)

        except:
            train_sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_label, img_box, crops = next(train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_label = cls_label.to(device, non_blocking=True)
        img_box = None

        # get local crops from uncertain regions
        cams, cams_aux = multi_scale_cam2(model, inputs=inputs, scales=args.cam_scales)
        cls,  segs, fmap, cls_aux, attn_pred, prob_out = model(inputs, training=True)

        # cls loss & aux cls loss
        cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
        cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

        # seg_loss & reg_loss
        valid_cam, pseudo_label = cam_to_label(cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label,  high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )
        segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        # ptc loss
        resized_cams_aux = F.interpolate(cams_aux, size=fmap.shape[2:], mode="bilinear", align_corners=False)
        _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        aff_mask = label_to_aff_mask(pseudo_label_aux)
        ptc_loss = get_masked_ptc_loss(fmap, aff_mask)

        # aff loss
        attn_mask = get_mask_by_radius(h=fmap.shape[-2],w=fmap.shape[-1],radius=args.radius)
        refined_aff_pseudo_label = refine_cams_with_bkg_aff(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label,  high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, aff_mat=attn_pred)
        final_seg_label = dr(n_iter,pseudo_label, refined_pseudo_label,refined_aff_pseudo_label)
        aff_label_ = F.interpolate(final_seg_label.clone().float().unsqueeze(1), size=fmap.shape[2:], mode="bilinear", align_corners=False).squeeze(1) if n_iter >= args.aux2final else pseudo_label_aux.clone().float()
        aff_label = cams_to_affinity_label(aff_label_, mask=attn_mask, ignore_index=args.ignore_index)
        aff_loss = get_aff_loss(attn_pred, aff_label)

        # kl loss
        KL_loss = KLloss(reduction='batchmean')
        confident_label =  final_seg_label.clone().float() if n_iter >= args.aux2final else pseudo_label_aux.clone().float()
        confident_label[confident_label==255]=0
        confident_label[confident_label>=1]=1
        dis_label = F.interpolate(confident_label.unsqueeze(1), size=fmap.shape[2:], mode='bilinear', align_corners=False).squeeze(1)
        kl_loss = KL_loss(prob_out,dis_label)

        seg_label =  final_seg_label if n_iter >= args.seg2final else refined_pseudo_label
        seg_loss = get_seg_loss(segs, seg_label.type(torch.long), ignore_index=args.ignore_index)
        reg_loss = get_energy_loss(img=inputs, logit=segs, label=seg_label, img_box=img_box, loss_layer=loss_layer)

        # warmup
        if n_iter <= 8000:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.0 * ptc_loss + 0.0 * aff_loss + args.w_kl * kl_loss + 0.0 * seg_loss + 0.0 * reg_loss
        elif n_iter <= 12000:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + 0.0 * aff_loss + args.w_kl * kl_loss + 0.0 * seg_loss + 0.0 * reg_loss
        elif n_iter <= 16000:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.w_aff * aff_loss + args.w_kl * kl_loss + 0.0 * seg_loss + 0.0 * reg_loss
        else:
            loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.w_aff * aff_loss + args.w_kl * kl_loss + args.w_seg * seg_loss + args.w_reg * reg_loss

        cls_pred = (cls > 0).type(torch.int16)
        cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])

        avg_meter.add({
            'cls_loss': cls_loss.item(),
            'ptc_loss': ptc_loss.item(),
            'aff_loss': aff_loss.item(),
            'kl_loss':  kl_loss.item(),
            'cls_loss_aux': cls_loss_aux.item(),
            'seg_loss': seg_loss.item(),
            'cls_score': cls_score.item(),
        })


        optim.zero_grad()
        loss.backward()
        optim.step()

        if (n_iter + 1) % args.log_iters == 0:
            delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
            cur_lr = optim.param_groups[0]['lr']

            if args.local_rank == 0:
                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, cls_loss_aux: %.4f, ptc_loss: %.4f, aff_loss: %.4f, kl_loss: %.4f, seg_loss: %.4f..." % (n_iter + 1, delta, eta, cur_lr, \
                                                        avg_meter.pop('cls_loss'), avg_meter.pop('cls_loss_aux'), avg_meter.pop('ptc_loss'), avg_meter.pop('aff_loss'), avg_meter.pop('kl_loss'), avg_meter.pop('seg_loss')))
                if args.tensorboard == True:
                    grid_img1, grid_cam1 = make_grid_image(inputs.detach(), cams.detach(), cls_label.detach())
                    _, grid_cam_aux = make_grid_image(inputs.detach(), cams_aux.detach(), cls_label.detach())
                    grid_seg_gt1 = make_grid_label(refined_pseudo_label.detach())
                    grid_seg_gt2 = make_grid_label(refined_aff_pseudo_label.detach())
                    grid_seg_pred = make_grid_label(torch.argmax(segs.detach(), dim=1))

                    tb_logger.add_image("visual/img1", grid_img1, global_step=global_step)
                    tb_logger.add_image("visual/cam1", grid_cam1, global_step=global_step)
                    tb_logger.add_image("visual/aux_cam", grid_cam_aux, global_step=global_step)
                    tb_logger.add_image("visual/seg_gt1", grid_seg_gt1, global_step=global_step)
                    tb_logger.add_image("visual/seg_gt_aff", grid_seg_gt2, global_step=global_step)
                    tb_logger.add_image("visual/seg_pred", grid_seg_pred, global_step=global_step)

        eval_flag = True if (n_iter + 1) >= args.begin_val or (n_iter + 1) % 10000 == 0 else False

        ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
        cur_lr = optim.param_groups[0]['lr']
        step_ = n_iter + 1

        if (n_iter + 1) % args.eval_iters == 0 and eval_flag:
            if args.local_rank == 0:
                logging.info('Validating...')
                if args.save_ckpt:
                    torch.save(

                        {
                            "step": step_,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optim.state_dict(),
                            "lr": cur_lr
                        },
                        ckpt_name
                        )
            val_cls_score, tab_results = validate(model=model, data_loader=val_loader, args=args)
            if args.local_rank == 0:
                logging.info("val cls score: %.6f" % (val_cls_score))
                logging.info("\n"+tab_results)


        if (n_iter + 1) % 1000 == 0:
            ckpt_name = os.path.join(args.ckpt_dir, "model_iter_latest.pth")
            if args.local_rank == 0:
                logging.info(f'Saving_{step_}_cpt...')
                if args.save_ckpt:
                    torch.save(

                        {
                            "step": step_,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optim.state_dict(),
                            "lr": cur_lr
                        },
                        ckpt_name
                        )

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    timestamp_1 = "{0:%Y-%m}".format(datetime.datetime.now())
    timestamp_2 = "{0:%d-%H-%M-%S}".format(datetime.datetime.now())
    exp_tag = f'{args.log_tag}_{timestamp_2}'
    
    if os.path.exists(args.resume_path) and args.resume_train:
        args.work_dir = args.resume_path.split('checkpoints')[0]
        args.log_dir = os.path.join(args.work_dir, f'train_resume_from_{timestamp_2}.log')
    else:
        args.work_dir = os.path.join(args.work_dir, 'coco', timestamp_1, exp_tag)
        args.log_dir = os.path.join(args.work_dir, f'train.log')

    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")
    args.tb_dir = os.path.join(args.work_dir, "tensorboards")

    if args.local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)
        os.makedirs(args.tb_dir, exist_ok=True)
        setup_logger(filename=args.log_dir)
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s"%(torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)

    ## fix random seed
    setup_seed(args.seed)
    train(args=args)