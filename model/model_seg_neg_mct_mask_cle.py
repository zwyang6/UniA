import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import backbone as encoder
from . import decoder
from .uncertainty_estimate import get_uncertainty

class network(nn.Module):
    def __init__(self, args, backbone, num_classes=None, pretrained=None, init_momentum=None, aux_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.init_momentum = init_momentum

        self.encoder = getattr(encoder, backbone)(pretrained=pretrained, aux_layer=aux_layer)
        
        self.in_channels = [self.encoder.embed_dim] * 4 if hasattr(self.encoder, "embed_dim") else [self.encoder.embed_dims[-1]] * 4 

        self.pooling = F.adaptive_max_pool2d

        if args.decoder == "largefov":
            self.decoder = decoder.LargeFOV(in_planes=self.in_channels[-1], out_planes=self.num_classes,)
        elif args.decoder == 'aspp':
            self.decoder = decoder.ASPP(in_planes=self.in_channels[-1], out_planes=self.num_classes,)   

        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes-1, kernel_size=1, bias=False,)
        self.aux_classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes-1, kernel_size=1, bias=False,)

        self.attn_proj = nn.Conv2d(in_channels=self.encoder.num_heads * 2, out_channels=1, kernel_size=1, bias=True)

        self.uncertainty_infer = get_uncertainty(self.encoder.embed_dim,16,pooling_size=args.pooling_size,hidden_dim=1024)


    def get_param_groups(self):

        param_groups = [[], [], [], []] # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[0].append(param)
            else:
                param_groups[1].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.aux_classifier.weight)
        param_groups[2].append(self.attn_proj.weight)
        for param in list(self.uncertainty_infer.parameters()):
            param_groups[2].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    def to_2D(self, x, h, w):
        n, hw, c = x.shape
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x
    
    def gen_attn_pred(self,attns):
        
        attn_cat = torch.cat(attns, dim=1)#.detach() # 4,12,400,400
        attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2)
        # attn_pred = attn_cat.mean(1).unsqueeze(1)
        attn_pred = self.attn_proj(attn_cat)
        attn_pred = torch.sigmoid(attn_pred)[:,0,...]

        return attn_pred

    def forward(self, x, cam_only=False,training=False):

        # cls_label = None
        _, _x, x_aux, attns = self.encoder.forward_features(x)
        h, w = x.shape[-2] // self.encoder.patch_size, x.shape[-1] // self.encoder.patch_size
        ## uncertrainty_infer
        _x4 ,prob_out = self.uncertainty_infer(_x)
        _x4 += _x4
        _x4 = self.to_2D(_x4, h, w)
        _x_aux = self.to_2D(x_aux, h, w)
        prob_out = self.to_2D(prob_out, h, w).mean(1)

        if cam_only:

            cam = F.conv2d(_x4, self.classifier.weight).detach()
            cam_aux = F.conv2d(_x_aux, self.aux_classifier.weight).detach()

            return cam_aux, cam, #attn_pred

        cls_aux = self.pooling(_x_aux, (1,1))
        cls_aux = self.aux_classifier(cls_aux)

        cls_x4 = self.pooling(_x4, (1,1))
        cls_x4 = self.classifier(cls_x4)

        cls_x4 = cls_x4.view(-1, self.num_classes-1)
        cls_aux = cls_aux.view(-1, self.num_classes-1)

        seg = self.decoder(_x4)
        attn_pred = self.gen_attn_pred(attns)  
        
        #! TO DO
        if training is False:
            return cls_x4, seg, _x4, cls_aux, attn_pred
        else:
            return cls_x4, seg, _x4, cls_aux, attn_pred, prob_out