def get_params_for_weight_decay(model, zero_wd_keys=[], wd=1e-4):
    n_wd, n_non_wd = [], []
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        
        if (any(nd in n for nd in zero_wd_keys)):
            n_non_wd.append(n)
            p_non_wd.append(p)
        else:
            n_wd.append(n)
            p_wd.append(p)
    print(f'Weight decay NOT applied to: {n_non_wd}')
    # print(f'Weight decay applied to: {n_wd}')
    return [
        {'params': p_wd, 'weight_decay': wd},
        {'params': p_non_wd, 'weight_decay': 0.0}
    ]
