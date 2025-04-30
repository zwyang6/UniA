import torch
import torch.nn as nn
from functools import partial
from model.backbone.vit_mask import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import torch.nn.functional as F


import math

default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'dino_base_patch8_224': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}


__all__ = [
    'deit_small_MCTformerV1_patch16_224', 'deit_small_MCTformerV2_patch16_224'
]

class MCTformerV1(VisionTransformer):
    def __init__(self, last_opt='average', pretrained_cfg=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_opt = last_opt
        if last_opt == 'fc':
            self.head = nn.Conv1d(in_channels=self.num_classes, out_channels=self.num_classes, kernel_size=self.embed_dim, groups=self.num_classes)
            self.head.apply(self._init_weights)

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_classes, self.embed_dim))
        # self.norm_token = partial(nn.LayerNorm, eps=1e-6)(self.embed_dim)
        
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        print(self.training)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
        N = self.pos_embed.shape[1] - self.num_classes
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:self.num_classes]
        patch_pos_embed = self.pos_embed[:, self.num_classes:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)
    
    def gen_attn_masks(self,x, cls_labels):
        _,n_cls = cls_labels.shape
        hw = x.shape[1] - n_cls
        b = x.shape[0]

        mask_cls_patch = cls_labels.unsqueeze(-1).repeat((b // cls_labels.shape[0]),1,hw)
        mask_cls_cls = torch.eye(n_cls).unsqueeze(0).repeat(b,1,1).to(cls_labels.device) # mask_cls_cls: bH*20*20

        attn_mask = torch.ones(b, (hw+n_cls), (hw+n_cls)).to(cls_labels.device) # attn_mask: bH*hw*hw+n_cls
        attn_mask[:,:n_cls,:n_cls] = mask_cls_cls
        attn_mask[:,:n_cls,n_cls:] = mask_cls_patch
        attn_mask[:,n_cls:,:n_cls] = mask_cls_patch.transpose(2,1)

        attn_mask = attn_mask.unsqueeze(1).repeat(1,self.num_heads,1,1).detach()

        return attn_mask

    def forward_features(self, x, cls_label, n=12):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)

        feats = []
        attn_weights = []
        if cls_label is not None:
            attn_mask = self.gen_attn_masks(x,cls_label)
        else:
            attn_mask = None
        for i, blk in enumerate(self.blocks):
            # x, weights_i = blk(x)
            x, weights_i = blk(x, attn_mask)
            feats.append(x)
            # if len(self.blocks) - i <= n:
            #     attn_weights.append(weights_i) 

        x = torch.cat([x[:, 0:self.num_classes],self.norm(x[:, self.num_classes:])], dim=1)
        # x = torch.cat([self.norm_token(x[:, 0:self.num_classes]),self.norm(x[:, self.num_classes:])], dim=1)
        feats[-1] = x

        return  x[:, 0:self.num_classes], x[:, self.num_classes:], feats[self.aux_layer][:, self.num_classes:]
                                                                                                                                                      
    def forward(self, x, cls_label, n_layers=12, return_att=False):
        x, attn_weights = self.forward_features(x,cls_label)

        attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N
        mtatt = attn_weights[-n_layers:].sum(0)[:, 0:self.num_classes, self.num_classes:]
        patch_attn = attn_weights[:, :, self.num_classes:, self.num_classes:]

        x_cls_logits = x.mean(-1)

        if return_att:
            return x_cls_logits, mtatt, patch_attn
        else:
            return x_cls_logits


@register_model
def deit_small_MCTformerV2_patch16_224(pretrained=False, **kwargs):
    model = MCTformerV2(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['cls_token', 'pos_embed']}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@register_model
def deit_small_MCTformerV1_patch16_224(pretrained=False, **kwargs):
    model = MCTformerV1(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])

    return model

@register_model
def deit_small_MCTformerV1_patch16_224(pretrained=False, **kwargs):
    model = MCTformerV1(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])

    return model
@register_model
def mctvit_mask_base_patch16_224(pretrained=False, **kwargs):
    model = MCTformerV1(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_224']
    if pretrained: 
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth",
            map_location="cpu", check_hash=True
        )
        # model.load_state_dict(checkpoint,strict=False)
        # load_pretrained(
        #     model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model