
import torch
import torch.nn.functional as F
import numpy as np
import math


def refine_cam_with_aff(valid_cam,aff_mat,cls_labels,par,inputs_denorm,ignore_index=None,high_thre=None,low_thre=None,radius=None):
    
    infer_size = int(math.sqrt(aff_mat.shape[-1]))
    valid_cam_resized = F.interpolate(valid_cam, size=(infer_size, infer_size), mode='bilinear', align_corners=False)
    attn_mask = get_mask_by_radius(h=infer_size,w=infer_size,radius=radius)
    aff_cam_l = propagte_aff_cam_with_bkg(valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask, cls_labels=cls_labels, bkg_score=low_thre)
    aff_cam_l = F.interpolate(aff_cam_l, size=valid_cam.shape[2:], mode='bilinear', align_corners=False)
    aff_cam_h = propagte_aff_cam_with_bkg(valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask, cls_labels=cls_labels, bkg_score=high_thre)
    aff_cam_h = F.interpolate(aff_cam_h, size=valid_cam.shape[2:], mode='bilinear', align_corners=False)

    bkg_cls = torch.ones(size=(valid_cam.shape[0], 1))
    bkg_cls = bkg_cls.to(valid_cam.device)
    _cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_aff_cam_l = refine_cams_with_cls_label(par, inputs_denorm, cams=aff_cam_l, labels=_cls_labels)
    refined_aff_label_l = refined_aff_cam_l.argmax(dim=1)
    refined_aff_cam_h = refine_cams_with_cls_label(par, inputs_denorm, cams=aff_cam_h, labels=_cls_labels)
    refined_aff_label_h = refined_aff_cam_h.argmax(dim=1)

    refined_aff_label = refined_aff_label_h.clone()
    refined_aff_label[refined_aff_label_h == 0] = ignore_index
    refined_aff_label[(refined_aff_label_h + refined_aff_label_l) == 0] = 0

    return refined_aff_label


def propagte_aff_cam_with_bkg(cams, aff=None, mask=None, cls_labels=None, bkg_score=None):

    b,_,h,w = cams.shape

    bkg = torch.ones(size=(b,1,h,w))*bkg_score
    bkg = bkg.to(cams.device)

    bkg_cls = torch.ones(size=(b,1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    cams_with_bkg = torch.cat((bkg, cams), dim=1)

    cams_rw = torch.zeros_like(cams_with_bkg)
    ##########
    b, c, h, w = cams_with_bkg.shape
    n_pow = 2
    n_log_iter = 0

    if mask is not None:
        for i in range(b):
            aff[i, mask==0] = 0

    aff = aff.detach() ** n_pow
    aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-1) ## avoid nan

    for i in range(n_log_iter):
        aff = torch.matmul(aff, aff)

    for i in range(b):
        _cams = cams_with_bkg[i].reshape(c, -1)
        valid_key = torch.nonzero(cls_labels[i,...])[:,0]
        _cams = _cams[valid_key,...]
        _cams = F.softmax(_cams, dim=0)
        _aff = aff[i]
        _cams_rw = torch.matmul(_cams, _aff)
        cams_rw[i, valid_key,:] = _cams_rw.reshape(-1, cams_rw.shape[2], cams_rw.shape[3])

    return cams_rw


def multi_scale_cam_with_aff(model, inputs, scales, cls_labels=None):
    cam_list, aff_mat = [], []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam, _cam_aux, _aff_mat = model(inputs_cat, cam_only=True)
        # aff_mat.append(_aff_mat)

        _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
        _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))
        
        cam_list = [F.relu(_cam)]
        cam_aux_list = [F.relu(_cam_aux)]


        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam, _cam_aux, _ = model(inputs_cat, cam_only=True)

                # aff_mat.append(_aff_mat)

                _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))

                cam_list.append(F.relu(_cam))
                cam_aux_list.append(F.relu(_cam_aux))


        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

    # max_aff_mat = aff_mat[np.argmax(scales)]
    return cam, cam_aux, _aff_mat


def refine_cams_with_cls_label(ref_mod=None, images=None, labels=None, cams=None):
    
    refined_cams = torch.zeros_like(cams)
    b = images.shape[0]

    #bg_label = torch.ones(size=(b, 1),).to(labels.device)
    cls_label = labels

    for idx in range(cls_label.shape[0]):

        _images = images[[idx], :, ...]

        _, _, h, w = _images.shape
        _images_ = F.interpolate(_images, size=[h//2, w//2], mode="bilinear", align_corners=False)

        valid_key = torch.nonzero(cls_label[idx,...])[:,0]
        valid_cams = cams[[idx], :, ...][:, valid_key,...]

        _refined_cams = ref_mod(_images_, valid_cams)
        _refined_cams = F.interpolate(_refined_cams, size=_images.shape[2:], mode="bilinear", align_corners=False)

        refined_cams[idx, valid_key, ...] = _refined_cams[0,...]

    return refined_cams

def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w 
    #_hw = (h + max(dilations)) * (w + max(dilations)) 
    mask  = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius+1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius+1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask


def refine_cams_with_bkg_aff(
    ref_mod=None,
    images=None,
    cams=None,
    cls_labels=None,
    high_thre=None,
    low_thre=None,
    ignore_index=False,
    img_box=None,
    down_scale=2,
    aff_mat=None,
):
    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b, 1, h, w)) * high_thre
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b, 1, h, w)) * low_thre
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * ignore_index
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()

    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    cams_with_bkg_h = F.interpolate(
    cams_with_bkg_h, size=[h // 16, w // 16], mode="bilinear", align_corners=False)  # .softmax(dim=1)
    cams_with_bkg_l = F.interpolate(
    cams_with_bkg_l, size=[h // 16, w // 16], mode="bilinear", align_corners=False)  # .softmax(dim=1)


    for idx in range(b):
        valid_key = torch.nonzero(cls_labels[idx, ...])[:, 0]
        valid_cams_h = cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        ## aff_refine
        _b,c,_h,_w = valid_cams_h.shape
        _aff = aff_mat[idx]
        _cams_aff_h = torch.matmul(valid_cams_h.reshape(_b,c,-1), _aff).reshape(_b,c,_h,_w)
        _cams_aff_l = torch.matmul(valid_cams_l.reshape(_b,c,-1), _aff).reshape(_b,c,_h,_w)

        valid_cams_h = F.interpolate(
        _cams_aff_h, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)  # .softmax(dim=1)
        valid_cams_l = F.interpolate(
        _cams_aff_l, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)  # .softmax(dim=1)

        _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_h, valid_key=valid_key, orig_size=(h, w))
        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_l, valid_key=valid_key, orig_size=(h, w))

        if img_box is not None:
            coord = img_box[idx]
            refined_label_h[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_h[0, coord[0]:coord[1], coord[2]:coord[3]]
            refined_label_l[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_l[0, coord[0]:coord[1], coord[2]:coord[3]]
        else:
            refined_label_h[idx] = _refined_label_h[0]
            refined_label_l[idx] = _refined_label_l[0]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = ignore_index
    refined_label[(refined_label_h + refined_label_l) == 0] = 0

    return refined_label


def _refine_cams(ref_mod, images, cams, valid_key, orig_size):
    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label


