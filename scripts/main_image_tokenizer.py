import argparse
from collections import OrderedDict
from omegaconf import OmegaConf
import os
import time
from einops import rearrange
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from timm.utils import ModelEmaV2

from transcoder.data.loader import InfiniteDataLoader
from transcoder.evaluations.fid.fid_score import calculate_frechet_distance
from transcoder.evaluations.lpips import get_lpips
from transcoder.evaluations.psnr import get_psnr
from transcoder.evaluations.ssim import get_ssim_and_msssim
from transcoder.optim.utils import get_params_for_weight_decay
from transcoder.utils import distributed as dist_utils
from transcoder.utils import config as config_utils
from transcoder.utils.meters import AverageMeter, ProgressMeter, SPSMeter
from transcoder.utils.misc import check_loss_nan, get_metrics_dict, get_grad_norm, tensor2image

import wandb


def get_args_parser():
    parser = argparse.ArgumentParser(description='Image Tokenizer', add_help=False)
    parser.add_argument('config', type=str, help='config')
    parser.add_argument('--output-dir', default='./', help='Output directory')
    parser.add_argument('--resume', default=None, type=str, help='checkpoint to resume')
    parser.add_argument('--eval-freq', default=20_000, type=int, help='evaluation frequency')
    parser.add_argument('--save-freq', default=1_000, type=int, help='save frequency')

     # EMA
    parser.add_argument('--use-ema', action='store_true', help='use exponential moving average')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='decay for EMA')
    parser.add_argument('--cpu-ema', action='store_true', help='put EMA weights on CPU')
    parser.add_argument('--ema-eval-freq', default=40_000, type=int, help='evaluation frequency for EMA')

    # system
    parser.add_argument('--start-iter', default=0, type=int, help='starting iteration')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--vis-freq', default=0, type=int, help='visualization frequency')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate', action='store_true', help='eval only')
    parser.add_argument('--skip-quantize', action='store_true', help='skip quantize at evaluation')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--no-wandb', action='store_true', help='disable wandb')
    return parser


def main(args):
    dist_utils.init_distributed_mode(args)
    dist_utils.random_seed(args.seed, dist_utils.get_rank())

    config = OmegaConf.load(args.config)
    print(config)
    print(args)

    if not args.no_wandb and config.get("wandb", None) is not None and dist_utils.is_main_process():
        run_name = f"{config.wandb.get('run', 'anonymous')}-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        try:
            wandb.init(
                # set the wandb project where this run will be logged
                project=config.wandb.get("project", "videocoding"),
                name=run_name,
                # track hyperparameters and run metadata
                config=OmegaConf.to_container(config, resolve=True),
                # save python code files
                settings=wandb.Settings(code_dir=".")
            )
            with_wandb = True
        except:
            print("Failed to initialize wandb")
            with_wandb = False
    else:
        print("Not using wandb; set wandb in the config to use.")
        with_wandb = False

    config.model.params.clamp_range = (-1, 1) if config.data.zero_mean else (0, 1)
    model = config_utils.instantiate_from_config(config.model)
    assert not config.data.zero_mean or not config.model.params.get('logit_laplace', False), "logit laplace mode is only compatible with the input being [0, 1]"
    model.cuda(args.gpu)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            bucket_cap_mb=200,
            find_unused_parameters=config.loss.params.disc_start > 0, # only used if the discriminator kicks in later
        )

    if args.use_ema:
        ema_model = ModelEmaV2(model, args.ema_decay, 
                               device=torch.cpu() if args.cpu_ema else None)        

    if "compile" in config:
        if config.compile.enable:
            model = torch.compile(model, options={"max_autotune": True, 
                                                  "epilogue_fusion": True, 
                                                  "trace.graph_diagram": True})
    # loss
    config.loss.params.use_bf16 = not config.optimizer.disable_amp
    config.loss.params.zero_mean = config.data.zero_mean
    criterion = config_utils.instantiate_from_config(config.loss).cuda(args.gpu)

    # optimizer
    config.optimizer.params.lr = config.optimizer.base_lr * args.world_size * config.data.batch_size 
    print(f"base LR is {config.optimizer.params.lr}")

    optimizer_generator = config_utils.instantiate_optimizer_from_config(
        config.optimizer,
        get_params_for_weight_decay(model, zero_wd_keys=config.optimizer.get("zero_wd_keys", []), wd=config.optimizer.params.weight_decay),
    )

    optimizer_discriminator = config_utils.instantiate_optimizer_from_config(
        config.optimizer,
        get_params_for_weight_decay(criterion.discriminator, zero_wd_keys=config.optimizer.get("zero_wd_keys", []), wd=config.optimizer.params.weight_decay),
        )
    enable_scaler = not config.optimizer.disable_amp and (not config.optimizer.get("use_bf16", False) or config.optimizer.get("force_bf16_scaler", False))
    scaler = amp.GradScaler(enabled=enable_scaler)

    # optionally resume from a checkpoint
    latest = os.path.join(args.output_dir, 'checkpoint.pt')
    if os.path.isfile(latest):
        args.resume = ''
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_iter = checkpoint['iter'] if 'iter' in checkpoint else 0
            result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            if args.use_ema:
                if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
                    ema_model.load_state_dict(checkpoint['ema_state_dict'], strict=False)
                else:
                    print("EMA speicifed but not found in the checkpoint, will reset EMA to the model state.")
                    ema_model.set(model)
            print(result)
            print(checkpoint['iter'])
            if args.start_iter >= config.loss.params.disc_start:
                result = criterion.discriminator.load_state_dict(checkpoint['state_dict_discriminator'], strict=True)
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
                print(result)

            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])        
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        latest = os.path.join(args.output_dir, 'checkpoint.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            args.start_iter = latest_checkpoint['iter']
            result = model.load_state_dict(latest_checkpoint['state_dict'])
            if args.use_ema:
                if 'ema_state_dict' in latest_checkpoint and latest_checkpoint['ema_state_dict'] is not None:
                    ema_model.load_state_dict(latest_checkpoint['ema_state_dict'])
                else:
                    print("EMA speicifed but not found in the checkpoint, will reset EMA to the model state.")
                    ema_model.set(model)
            print(result)
            if args.start_iter >= config.loss.params.disc_start:
                result = criterion.discriminator.load_state_dict(latest_checkpoint['state_dict_discriminator'], strict=True)
                optimizer_discriminator.load_state_dict(latest_checkpoint['optimizer_discriminator'])
                print(result)

            optimizer_generator.load_state_dict(latest_checkpoint['optimizer_generator'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            print("=> loaded latest checkpoint '{}' (iter {})"
                  .format(latest, latest_checkpoint['iter']))

    image_size = config.data.image_size

    if config.data.get('use_rrc', False):
        trainTs = [
            transforms.RandomResizedCrop(image_size),
        ]
    else:
        trainTs = [
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size),
        ]
    if config.data.get('use_hflip', False):
        trainTs.append(transforms.RandomHorizontalFlip(0.5))
    trainTs += [transforms.ToTensor()]
    interpolation = config.evaluation.get('interpolation', 'bilinear')
    if interpolation == "bilinear":
        interpolation = transforms.InterpolationMode.BILINEAR
    elif interpolation == 'lanczos':
        interpolation = transforms.InterpolationMode.LANCZOS
    else:
        raise ValueError
    valTs = [
        transforms.Resize(image_size, interpolation=interpolation),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    if config.data.zero_mean:
        trainTs.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        valTs.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    train_transform = transforms.Compose(trainTs)
    val_transform = transforms.Compose(valTs)
    train_dataset = config_utils.instantiate_dataset_from_config(config.data.train, train_transform)
    valid_dataset = config_utils.instantiate_dataset_from_config(config.data.val, val_transform)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False)
    else:
        train_sampler = None
        valid_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=(train_sampler is None),
        collate_fn=None,
        num_workers=config.data.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    print(f"{len(train_loader)=}")
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.data.batch_size,
        shuffle=(valid_sampler is None),
        collate_fn=None,
        num_workers=config.data.num_workers,
        pin_memory=False,
        sampler=valid_sampler,
        drop_last=False,
    )
    print(f"{len(valid_loader)=}")
    train_loader = InfiniteDataLoader(train_loader)

    lr_scheduler = (
        config_utils.instantiate_from_config(config.optimizer.lr_scheduler_config)
        if config.optimizer.get("lr_scheduler_config", None) is not None else None
    )

    if args.evaluate:
        evaluate(valid_loader, model, args, config, full_eval=True)
        if args.use_ema:
            print('running ema model evaluation')
            evaluate(valid_loader, ema_model.module, args, config, full_eval=True)
        return
    
    iter = args.start_iter
    batch_time = AverageMeter('Time', ':6.2f')
    sps_meter = SPSMeter("Img/Sec")
    h_meter = AverageMeter("H", ":6.2f")
    data_time = AverageMeter('Data', ':6.2f')
    model_time = AverageMeter('Model', ':6.2f')
    grad_norm = AverageMeter('GradNorm', ':6.2f')
    loss_scale = AverageMeter('LossScale', ':6.2f')
    lr = AverageMeter('LR', ':.2e')
    mem = AverageMeter('Mem', ':6.1f')
    metric_names = [
        'train/total_loss', 'train/quant_loss', 'train/nll_loss', 'train/rec_loss',
        'train/p_loss', 'train/d_weight', 'train/disc_factor', 'train/g_loss',
        'train/disc_loss', 'train/logits_real', 'train/logits_fake',
        'train/disc_r1_loss', 'train/disc_r1_loss_scale',
    ]
    if config.loss.get('logit_laplace_weight', 0) > 0:
        assert dist_utils.get_model(model).logit_laplace, "The model has to be compatible with logit laplace loss"
        metric_names.append('train/logit_laplace_loss')
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        config.optimizer.max_iter,
        [lr, batch_time, sps_meter, data_time, model_time, mem, h_meter, grad_norm, loss_scale, *metrics.values()],
        prefix="iter: ",
    )
    best_fid = np.Inf
    while True:
        model.train()
        criterion.discriminator.train()
        end = time.time()
        for data, _ in train_loader:
            data_time.update(time.time() - end)

            # update weight decay and learning rate according to their schedule
            if lr_scheduler is not None:
                for k, param_group in enumerate(optimizer_discriminator.param_groups):
                    param_group['lr'] = config.optimizer.params.lr * lr_scheduler(iter)
                for k, param_group in enumerate(optimizer_generator.param_groups):
                    param_group['lr'] = config.optimizer.params.lr * lr_scheduler(iter)
                lr.update(config.optimizer.params.lr * lr_scheduler(iter))
            else:
                lr.update(optimizer_generator.param_groups[0]['lr'])

            x = data.cuda(args.gpu, non_blocking=True)
            with amp.autocast(enabled=not config.optimizer.disable_amp, dtype=torch.bfloat16 if config.optimizer.use_bf16 else torch.float16):
                x_recon, qloss, qinfo = model(x)

            skip_disc = (iter < config.loss.params.disc_start and config.loss.params.get("skip_disc", False))

            for optimizer_idx in range(2 - skip_disc):
                if optimizer_idx == 0:
                    # autoencoder
                    optimizer_generator.zero_grad()
                else:
                    # discriminator
                    optimizer_discriminator.zero_grad()
                tic = time.time()
                with amp.autocast(enabled=not config.optimizer.disable_amp, dtype=torch.bfloat16 if config.optimizer.use_bf16 else torch.float16):
                    loss, log = criterion(qloss, x, x_recon, optimizer_idx, iter, dist_utils.get_model(model).get_last_layer(), None, "train")
                    if 'logit_laplace_loss' in qinfo and optimizer_idx == 0:
                        loss = loss + qinfo['logit_laplace_loss'] * config.loss.logit_laplace_weight
                        log['train/logit_laplace_loss'] = qinfo['logit_laplace_loss']
                check_loss_nan(loss)
                scaler.scale(loss).backward()
                if iter >= config.loss.params.disc_start or not skip_disc:
                    scaler.step(optimizer_generator if optimizer_idx == 0 else optimizer_discriminator)
                    if args.use_ema and optimizer_idx == 0:
                        ema_model.update(model)
                else:
                    scaler.step(optimizer_generator) # before the discriminator kicks, we must only update the generator
                    if args.use_ema:
                        ema_model.update(model)
                scaler.update()

                for k, v in log.items():
                    if k not in metrics:
                        raise ValueError(f"Unknown metric {k}")
                    metrics[k].update(v.item(), config.data.batch_size)

                # only update meter at the end of each iteration
                if optimizer_idx == 1 - skip_disc:
                    model_time.update(time.time() - tic)
                    batch_time.update(time.time() - end)
                    sps_meter.update(time.time() - end, config.data.batch_size * args.world_size)
                    h_meter.update(qinfo["H"])
                    grad_norm.update(get_grad_norm(model))
                    loss_scale.update(scaler.get_scale())

                    end = time.time()
                    mem.update(torch.cuda.max_memory_allocated() // 1e9)

                    

            iter += 1
            if with_wandb:
                info = get_metrics_dict(iter, metrics)
                info.update({
                    'state/H': h_meter.val,
                    'state/grad_norm': grad_norm.val,
                    'state/loss_scale': loss_scale.val,
                    'state/samples_per_sec': sps_meter.avg,
                    'lr': lr.val,
                })
                if args.vis_freq > 0 and iter % args.vis_freq == 0:
                    num_rows = int(np.sqrt(x.shape[0]))
                    num_rows = min(num_rows, 4)    # avoid eating up too much memory
                    image = tensor2image(rearrange(x[:num_rows ** 2].detach(), '(r c) ch h w -> (r h) (c w) ch', r=num_rows), zero_mean=config.data.zero_mean)
                    recon = tensor2image(rearrange(x_recon[:num_rows ** 2].detach(), '(r c) ch h w -> (r h) (c w) ch', r=num_rows), zero_mean=config.data.zero_mean)         
                    info.update({
                        'images': [wandb.Image(image, caption='images (input)')],
                        'recon': [wandb.Image(recon, caption='images (recon)')]
                    })
                wandb.log(info, step=iter)

            if iter % args.print_freq == 0:
                progress.display(iter)
            if iter % args.eval_freq == 0:
                metrics_val = evaluate(valid_loader, model, args, config)
                dist_utils.save_on_master({
                    'iter': iter,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict() if args.use_ema else None,
                    'optimizer_generator' : optimizer_generator.state_dict() if dist_utils.is_main_process() else None,
                    'optimizer_discriminator' : optimizer_discriminator.state_dict() if dist_utils.is_main_process() and iter >= config.loss.params.disc_start else None,
                    'state_dict_discriminator': criterion.discriminator.state_dict() if iter >= config.loss.params.disc_start else None,
                    'scaler': scaler.state_dict(),
                    'config': config,
                    'args': args,
                    'best_fid': metrics_val['fid'],
                }, best_fid > metrics_val['fid'], args.output_dir)
                best_fid = min(best_fid, metrics_val['fid'])
                if with_wandb:
                    for k, v in metrics_val.items():
                        wandb.log({f'eval/{k}': v}, step=iter)
            if iter % args.save_freq == 0:
                dist_utils.save_on_master({
                    'iter': iter,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict() if args.use_ema else None,
                    'optimizer_generator' : optimizer_generator.state_dict() if dist_utils.is_main_process() else None,
                    'optimizer_discriminator' : optimizer_discriminator.state_dict() if dist_utils.is_main_process() and iter >= config.loss.params.disc_start else None,
                    'state_dict_discriminator': criterion.discriminator.state_dict() if iter >= config.loss.params.disc_start else None,
                    'scaler': scaler.state_dict(),
                    'config': config,
                    'args': args,
                }, False, args.output_dir)
            if args.use_ema and iter %  args.ema_eval_freq == 0:
                metrics_val = evaluate(valid_loader, ema_model.module, args, config)
                if with_wandb:
                    for k, v in metrics_val.items():
                        wandb.log({f'ema_eval/{k}': v}, step=iter)
            if iter >= config.optimizer.max_iter:
                stop = True
                break
        if stop:
            break
    progress.synchronize()


def evaluate(data_loader, model, args, config, full_eval=False):
    from transcoder.evaluations.fid.inception import InceptionV3
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[config.evaluation.fid.dims]
    inception_v3 = InceptionV3([block_idx], normalize_input=not config.data.zero_mean).cuda(args.gpu)
    if os.path.exists(config.evaluation.fid.groundtruth_npz):
        gt_cached = True
    else:
        gt_cached = False

    all_pred_x = [[] for _ in range(args.world_size)]
    all_pred_xr = [[] for _ in range(args.world_size)]

    if full_eval:
        all_psnr = [[] for _ in range(args.world_size)]
        all_ssim = [[] for _ in range(args.world_size)]
        all_msssim = [[] for _ in range(args.world_size)]
        all_lpips = [[] for _ in range(args.world_size)]
    total_num = 0

    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem', ':6.1f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, mem],
        prefix="iter: ",
    )
    end = time.time()
    model.eval()
    inception_v3.eval()
    all_codes = None

    with torch.inference_mode():
        
        for iter, (data, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            x = data.cuda(args.gpu, non_blocking=True)
            with amp.autocast(enabled=not config.optimizer.disable_amp, dtype=torch.bfloat16 if config.optimizer.use_bf16 else torch.float16):
                x_recon, _, info = model(x, skip_quantize=args.skip_quantize)
            if 'used_codes' in info:
                if all_codes is None:
                    all_codes = info['used_codes']
                else:
                    all_codes = torch.cat([all_codes, info['used_codes']]).unique()
            x_recon = x_recon.to(torch.float32)
            if full_eval:
                # PSNR
                pred_psnr = get_psnr(x, x_recon, zero_mean=config.data.zero_mean)
                gathered_psnr = [torch.zeros_like(pred_psnr) for _ in range(args.world_size)]
                torch.distributed.all_gather(gathered_psnr, pred_psnr)
                for j in range(args.world_size):
                    all_psnr[j].append(gathered_psnr[j].detach().cpu())
                # SSIM
                pred_ssim, pred_msssim = get_ssim_and_msssim(x, x_recon, zero_mean=config.data.zero_mean)
                gathered_ssim = [torch.zeros_like(pred_ssim) for _ in range(args.world_size)]
                gathered_msssim = [torch.zeros_like(pred_msssim) for _ in range(args.world_size)]
                torch.distributed.all_gather(gathered_ssim, pred_ssim)
                torch.distributed.all_gather(gathered_msssim, pred_msssim)
                for j in range(args.world_size):
                    all_ssim[j].append(gathered_ssim[j].detach().cpu())
                    all_msssim[j].append(gathered_msssim[j].detach().cpu())
                # LPIPS (AlexNet)
                pred_lpips = get_lpips(x, x_recon, zero_mean=config.data.zero_mean, network_type='alex')
                gathered_lpips = [torch.zeros_like(pred_lpips) for _ in range(args.world_size)]
                torch.distributed.all_gather(gathered_lpips, pred_lpips)
                for j in range(args.world_size):
                    all_lpips[j].append(gathered_lpips[j].detach().cpu())
            # FID
            if not gt_cached:
                pred_x = inception_v3(x)[0]
                if pred_x.size(2) != 1 or pred_x.size(3) != 1:
                    pred_x = F.adaptive_avg_pool2d(pred_x, (1, 1))
                pred_x = pred_x.squeeze(3).squeeze(2)
                gathered_pred_x = [torch.zeros_like(pred_x) for _ in range(args.world_size)]
                torch.distributed.all_gather(gathered_pred_x, pred_x)
                for j in range(args.world_size):
                    all_pred_x[j].append(gathered_pred_x[j].detach().cpu())

            pred_xr = inception_v3(x_recon)[0]
            if pred_xr.size(2) != 1 or pred_xr.size(3) != 1:
                pred_xr = F.adaptive_avg_pool2d(pred_xr, (1, 1))
            pred_xr = pred_xr.squeeze(3).squeeze(2)
            gathered_pred_xr = [torch.zeros_like(pred_xr) for _ in range(args.world_size)]
            torch.distributed.all_gather(gathered_pred_xr, pred_xr)
            for j in range(args.world_size):
                all_pred_xr[j].append(gathered_pred_xr[j].detach().cpu())
            total_num += args.world_size * x.shape[0]
            batch_time.update(time.time() - end)
            end = time.time()

            mem.update(torch.cuda.max_memory_allocated() // 1e9)
            if iter % args.print_freq == 0:
                progress.display(iter)
    progress.synchronize()
    if full_eval:
        # PSNR
        for j in range(args.world_size):
            all_psnr[j] = torch.cat(all_psnr[j], dim=0).numpy()
        all_psnr_reorg = []
        for j in range(total_num):
            all_psnr_reorg.append(all_psnr[j % args.world_size][j // args.world_size])
        all_psnr = np.vstack(all_psnr_reorg)[:config.evaluation.fid.num_samples]
        print(f"PSNR: {np.mean(all_psnr):.4f} (±{np.std(all_psnr):.4f})")
        # SSIM
        for j in range(args.world_size):
            all_ssim[j] = torch.cat(all_ssim[j], dim=0).numpy()
            all_msssim[j] = torch.cat(all_msssim[j], dim=0).numpy()
        all_ssim_reorg = []
        all_msssim_reorg = []
        for j in range(total_num):
            all_ssim_reorg.append(all_ssim[j % args.world_size][j // args.world_size])
            all_msssim_reorg.append(all_msssim[j % args.world_size][j // args.world_size])
        all_ssim = np.vstack(all_ssim_reorg)[:config.evaluation.fid.num_samples]
        all_msssim = np.vstack(all_msssim_reorg)[:config.evaluation.fid.num_samples]
        print(f"SSIM: {np.mean(all_ssim):.4f} (±{np.std(all_ssim):.4f})")
        print(f"MS-SSIM: {np.mean(all_msssim):.4f} (±{np.std(all_msssim):.4f})")
        # LPIPS
        for j in range(args.world_size):
            all_lpips[j] = torch.cat(all_lpips[j], dim=0).numpy()
        all_lpips_reorg = []
        for j in range(total_num):
            all_lpips_reorg.append(all_lpips[j % args.world_size][j // args.world_size])
        all_lpips = np.vstack(all_lpips_reorg)[:config.evaluation.fid.num_samples]
        print(f"LPIPS (AlexNet): {np.mean(all_lpips):.4f} (±{np.std(all_lpips):.4f})")
    
    # FID
    if not gt_cached:
        for j in range(args.world_size):
            all_pred_x[j] = torch.cat(all_pred_x[j], dim=0).numpy()
        all_pred_x_reorg = []
        for j in range(total_num):
            all_pred_x_reorg.append(all_pred_x[j % args.world_size][j // args.world_size])
        all_pred_x = np.vstack(all_pred_x_reorg)
        all_pred_x = all_pred_x[:config.evaluation.fid.num_samples]
        m2, s2 = np.mean(all_pred_x, axis=0), np.cov(all_pred_x, rowvar=False)
        np.savez_compressed(config.evaluation.fid.groundtruth_npz, mu=m2, sigma=s2)
    else:
        with np.load(config.evaluation.fid.groundtruth_npz) as f:
            m2, s2 = f['mu'][:], f['sigma'][:]
    for j in range(args.world_size):
        all_pred_xr[j] = torch.cat(all_pred_xr[j], dim=0).numpy()
    all_pred_xr_reorg = []
    for j in range(total_num):
        all_pred_xr_reorg.append(all_pred_xr[j % args.world_size][j // args.world_size])
    all_pred_xr = np.vstack(all_pred_xr_reorg)
    all_pred_xr = all_pred_xr[:config.evaluation.fid.num_samples]
    m1, s1 = np.mean(all_pred_xr, axis=0), np.cov(all_pred_xr, rowvar=False)
    fid_score = calculate_frechet_distance(m1, s1, m2, s2)
    print(f"FID: {fid_score:.4f}")

    # recover model to train mode
    model.train()

    if full_eval:
        ret =  {
            'psnr': np.mean(all_psnr),
            'ssim': np.mean(all_ssim),
            'ms-ssim': np.mean(all_msssim),
            'lpips-alexnet': np.mean(all_lpips),
            'fid': fid_score,
        }
    else:
        ret =  {'fid': fid_score}

    # cb usage
    if all_codes is not None:
        usage_rate = all_codes.size(0) / (2 ** config.model.params.embed_dim)
        print(f"Codebook usage: {all_codes.size(0)} / {2 ** config.model.params.embed_dim} ({usage_rate * 100:.02f}%)")
        ret['usage-rate'] =  usage_rate

    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Tokenizer', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
