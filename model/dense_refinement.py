def dr(n_iter,pseudo_label, refined_pseudo_label,refined_aff_label,lf_start_iters=6000):

    """fuse refined_aff_label and refined_pseudo_label"""
    # 使用refined_pseudo_label 优化refined_aff_label的边界信息以及内容信息
    # 即前者的12,34 等cls信息能填充后者的背景0 以及255 忽略的区域， 前者的255 不能能覆盖后者的背景0
    tamp_mask = (refined_pseudo_label != 0) * ((refined_pseudo_label != 255))
    bkg_mask  = (refined_aff_label != 0) * (refined_aff_label != 255)
    tamp_info_ = tamp_mask * refined_pseudo_label
    bkg_info_  = (bkg_mask * refined_aff_label) == 0
    tamp_info = tamp_info_ * bkg_info_
    final_seg_label_ = tamp_info  + (tamp_info == 0) * refined_aff_label
    
    if n_iter <= lf_start_iters:
        final_seg_label = final_seg_label_
    else:
        prior = (pseudo_label != 0) * (pseudo_label != 255)
        prior_mask = prior * (refined_pseudo_label == pseudo_label) 
        prior_info = prior_mask * pseudo_label

        final_seg_label = (prior_mask ==0)*final_seg_label_ + prior_info
        
    return final_seg_label

def dr_acdc(n_iter,pseudo_label, refined_pseudo_label,refined_aff_label,lf_start_iters=6000,aux_start_iters=9000):

    """fuse refined_aff_label and refined_pseudo_label"""
    # 使用refined_pseudo_label 优化refined_aff_label的边界信息以及内容信息
    # 即前者的12,34 等cls信息能填充后者的背景0 以及255 忽略的区域， 前者的255 不能能覆盖后者的背景0
    tamp_mask = (refined_pseudo_label != 0) * ((refined_pseudo_label != 255))
    bkg_mask  = (refined_aff_label != 0) * (refined_aff_label != 255)
    tamp_info_ = tamp_mask * refined_pseudo_label
    bkg_info_  = (bkg_mask * refined_aff_label) == 0
    tamp_info = tamp_info_ * bkg_info_
    final_seg_label_ = tamp_info  + (tamp_info == 0) * refined_aff_label
    
    if n_iter <= lf_start_iters:
        final_seg_label = final_seg_label_
    if n_iter > lf_start_iters:
        prior = (pseudo_label != 0) * (pseudo_label != 255)
        prior_mask = prior * (refined_pseudo_label == pseudo_label) 
        prior_info = prior_mask * pseudo_label

        final_seg_label = (prior_mask ==0)*final_seg_label_ + prior_info
    # 使用pseudo_label 进一步优化refined_aff_label的内容信息
    # 即前者的12,34 等cls信息能填充后者的255 忽略的区域，但是无法填充背景
    ## 考虑给pseudo_label 加一个random mask 除去一定噪声
    if n_iter >= aux_start_iters:
        ctx_mask255 = refined_pseudo_label == 255 
        ctx_info = ctx_mask255 * pseudo_label
        ctx_info_ignore255 = ctx_info * (ctx_info != 255) * (ctx_info != 0)
        # random_mask = torch.randn(ctx_info_ignore255.size()) >= 0.
        # random_mask = random_mask.cuda()
        # # random mask  
        # ctx_info_ignore255 = ctx_info_ignore255 * random_mask
        final_seg_label = ctx_info_ignore255 + (ctx_info_ignore255 == 0) * final_seg_label
    return final_seg_label

def dr_visual(pseudo_label, refined_pseudo_label,refined_aff_label):

    """fuse refined_aff_label and refined_pseudo_label"""
    # 使用refined_pseudo_label 优化refined_aff_label的边界信息以及内容信息
    # 即前者的12,34 等cls信息能填充后者的背景0 以及255 忽略的区域， 前者的255 不能能覆盖后者的背景0
    tamp_mask = (refined_pseudo_label != 0) * ((refined_pseudo_label != 255))
    bkg_mask  = (refined_aff_label != 0) * (refined_aff_label != 255)
    tamp_info_ = tamp_mask * refined_pseudo_label
    bkg_info_  = (bkg_mask * refined_aff_label) == 0
    tamp_info = tamp_info_ * bkg_info_
    final_seg_label_ = tamp_info  + (tamp_info == 0) * refined_aff_label
    

    prior = (pseudo_label != 0) * (pseudo_label != 255)
    prior_mask = prior * (refined_pseudo_label == pseudo_label) 
    prior_info = prior_mask * pseudo_label

    final_seg_label = (prior_mask ==0)*final_seg_label_ + prior_info

    return final_seg_label

    