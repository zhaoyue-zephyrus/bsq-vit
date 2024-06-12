import argparse
import os
import time

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as transforms

from transcoder.evaluations.fid.fid_score import calculate_frechet_distance
from transcoder.evaluations.lpips import get_lpips
from transcoder.evaluations.psnr import get_psnr
from transcoder.evaluations.ssim import get_ssim_and_msssim
from transcoder.tokenizers.dall_e.dalle import Dalle
from transcoder.tokenizers.sd import SD2Tokenizer, SDXLTokenizer
from transcoder.tokenizers.vqgan.vqgan import VQGAN
from transcoder.utils import distributed as dist_utils
from transcoder.utils.meters import AverageMeter, ProgressMeter


def get_args_parser():
    parser = argparse.ArgumentParser(description='Evaluate public tokenizer', add_help=False)
    parser.add_argument('--model', default='dalle-dvae', 
                        choices=['dalle-dvae', 'vqgan1024', 'vqgan16384', 'vqgan-kl', 'sdxl', 'sd2'],
                        type=str)
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--imagenet-val-path', default='/storage/Datasets/ILSVRC2012/val/', type=str, help='path to imagenet-1k val')
    parser.add_argument('--coco-val-path', default='/storage/Datasets/ILSVRC2012/coco/', type=str, help='path to coco val')
    parser.add_argument('--interpolation', default=None, type=str, choices=['lanczos'], help='interpolation method')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers per process')
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
    parser.add_argument('--save-to-folder', default=None, type=str, help='save results to folder')
    return parser


def main(args):
    dist_utils.init_distributed_mode(args)
    dist_utils.random_seed(args.seed, dist_utils.get_rank())

    if args.interpolation == 'lanczos':
        interpolation = transforms.InterpolationMode.LANCZOS
        imagenet_npz = 'imagenet_val_256x256_lanczos.npz'
        coco_npz = 'coco_val2017_256x256_lanczos.npz'
    else:
        interpolation = transforms.InterpolationMode.BILINEAR
        imagenet_npz = 'imagenet_val_256x256.npz'
        coco_npz = 'coco_val2017_256x256.npz'

    if args.model == 'dalle-dvae':
        model = Dalle(model_rootpath='./vqgan/tokenizers/dall_e/')
        val_transform = transforms.Compose([
            transforms.Resize(256, interpolation=interpolation),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
        args.zero_mean = False
    elif args.model in ['vqgan1024', 'vqgan16384', 'vqgan-kl', 'sdxl', 'sd2']:
        val_transform = transforms.Compose([
            transforms.Resize(256, interpolation=interpolation),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        if args.model == 'vqgan1024':
            model = VQGAN('third_party/vqgan/vq-f8-n256/config.yaml', 'third_party/vqgan/vq-f8-n256/model.ckpt')
        elif args.model == 'vqgan16384':
            model = VQGAN('third_party/vqgan/vq-f8/config.yaml', 'third_party/vqgan/vq-f8/model.ckpt')
        elif args.model == 'vqgan-kl':
            model = VQGAN('third_party/vqgan/vqgan_gumbel_f8/configs/model.yaml', 'third_party/vqgan/vqgan_gumbel_f8/checkpoints/last.ckpt', is_gumbel=True)
        elif args.model == 'sdxl':
            model = SDXLTokenizer()
        elif args.model == 'sd2':
            model = SD2Tokenizer()
        args.zero_mean = True
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model.cuda(args.gpu)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200)

    imagenet_dataset = torchvision.datasets.ImageFolder(args.imagenet_val_path, transform=val_transform)
    coco_dataset = torchvision.datasets.ImageFolder(args.coco_val_path, transform=val_transform)
    if args.distributed:
        imagenet_sampler = torch.utils.data.distributed.DistributedSampler(imagenet_dataset, shuffle=False)
        coco_sampler = torch.utils.data.distributed.DistributedSampler(coco_dataset, shuffle=False)
    else:
        imagenet_sampler = None
        coco_sampler = None
    imagenet_loader = torch.utils.data.DataLoader(
        imagenet_dataset,
        batch_size=args.batch_size,
        shuffle=(imagenet_sampler is None),
        collate_fn=None,
        num_workers=args.workers,
        pin_memory=False,
        sampler=imagenet_sampler,
        drop_last=False,
    )
    coco_loader = torch.utils.data.DataLoader(
        coco_dataset,
        batch_size=args.batch_size,
        shuffle=(coco_sampler is None),
        collate_fn=None,
        num_workers=args.workers,
        pin_memory=False,
        sampler=coco_sampler,
        drop_last=False,
    )
    evaluate(coco_loader, model, args, full_eval=True, num_samples=5_000, groundtruth_npz=coco_npz)
    evaluate(imagenet_loader, model, args, full_eval=True, groundtruth_npz=imagenet_npz)
    return

    
def evaluate(data_loader, model, args, full_eval=False, num_samples=50_000, groundtruth_npz='imagenet_val_256x256.npz'):
    from transcoder.evaluations.fid.inception import InceptionV3
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception_v3 = InceptionV3([block_idx], normalize_input=not args.zero_mean).cuda(args.gpu)
    if os.path.exists(groundtruth_npz):
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

    with torch.inference_mode():
        
        for iter, (data, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            x = data.cuda(args.gpu, non_blocking=True)
            x_recon = model(x)
            x_recon = x_recon.to(torch.float32)
            if full_eval:
                # PSNR
                pred_psnr = get_psnr(x, x_recon, zero_mean=args.zero_mean)
                gathered_psnr = [torch.zeros_like(pred_psnr) for _ in range(args.world_size)]
                torch.distributed.all_gather(gathered_psnr, pred_psnr)
                for j in range(args.world_size):
                    all_psnr[j].append(gathered_psnr[j].detach().cpu())
                # SSIM
                pred_ssim, pred_msssim = get_ssim_and_msssim(x, x_recon, zero_mean=args.zero_mean)
                gathered_ssim = [torch.zeros_like(pred_ssim) for _ in range(args.world_size)]
                gathered_msssim = [torch.zeros_like(pred_msssim) for _ in range(args.world_size)]
                torch.distributed.all_gather(gathered_ssim, pred_ssim)
                torch.distributed.all_gather(gathered_msssim, pred_msssim)
                for j in range(args.world_size):
                    all_ssim[j].append(gathered_ssim[j].detach().cpu())
                    all_msssim[j].append(gathered_msssim[j].detach().cpu())
                # LPIPS (AlexNet)
                pred_lpips = get_lpips(x, x_recon, zero_mean=args.zero_mean, network_type='alex')
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
        all_psnr = np.vstack(all_psnr_reorg)[:num_samples]
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
        all_ssim = np.vstack(all_ssim_reorg)[:num_samples]
        all_msssim = np.vstack(all_msssim_reorg)[:num_samples]
        print(f"SSIM: {np.mean(all_ssim):.4f} (±{np.std(all_ssim):.4f})")
        print(f"MS-SSIM: {np.mean(all_msssim):.4f} (±{np.std(all_msssim):.4f})")
        # LPIPS
        for j in range(args.world_size):
            all_lpips[j] = torch.cat(all_lpips[j], dim=0).numpy()
        all_lpips_reorg = []
        for j in range(total_num):
            all_lpips_reorg.append(all_lpips[j % args.world_size][j // args.world_size])
        all_lpips = np.vstack(all_lpips_reorg)[:num_samples]
        print(f"LPIPS (AlexNet): {np.mean(all_lpips):.4f} (±{np.std(all_lpips):.4f})")
    
    # FID
    if not gt_cached:
        for j in range(args.world_size):
            all_pred_x[j] = torch.cat(all_pred_x[j], dim=0).numpy()
        all_pred_x_reorg = []
        for j in range(total_num):
            all_pred_x_reorg.append(all_pred_x[j % args.world_size][j // args.world_size])
        all_pred_x = np.vstack(all_pred_x_reorg)
        all_pred_x = all_pred_x[:num_samples]
        m2, s2 = np.mean(all_pred_x, axis=0), np.cov(all_pred_x, rowvar=False)
        np.savez_compressed(groundtruth_npz, mu=m2, sigma=s2)
    else:
        with np.load(groundtruth_npz) as f:
            m2, s2 = f['mu'][:], f['sigma'][:]
    for j in range(args.world_size):
        all_pred_xr[j] = torch.cat(all_pred_xr[j], dim=0).numpy()
    all_pred_xr_reorg = []
    for j in range(total_num):
        all_pred_xr_reorg.append(all_pred_xr[j % args.world_size][j // args.world_size])
    all_pred_xr = np.vstack(all_pred_xr_reorg)
    all_pred_xr = all_pred_xr[:num_samples]
    m1, s1 = np.mean(all_pred_xr, axis=0), np.cov(all_pred_xr, rowvar=False)
    fid_score = calculate_frechet_distance(m1, s1, m2, s2)
    print(f"FID: {fid_score:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate public tokenizer', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
