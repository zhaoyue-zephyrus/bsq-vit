import torch
import transcoder.utils.distributed as dist_utils


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + 1e-16)

    def synchronize(self):
        if not dist_utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / (self.count + 1e-16)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
class SPSMeter:
    def __init__(self, name) -> None:
        self.name = name
        self.reset()
    
    def reset(self):
        self.last_n = 0
        self.last_t = 0
        self.total_n = 0
        self.total_t = 0
    
    def update(self, t, n):
        self.last_n = n
        self.last_t = t
        self.last = n / (t + 1e-16)
        self.total_n += n
        self.total_t += t
        self.avg = self.total_n / (self.total_t + 1e-16)
    
    def synchronize(self):
        if not dist_utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.total_n, self.total_t], dtype=torch.float64, device='cuda')
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.total_n= int(t[0])
        self.total_t = t[1] / torch.distributed.get_world_size()
        self.avg = self.total_n / (self.total_t + 1e-16)
    
    def __repr__(self) -> str:
        return f"{self.name} {self.last:.02f} ({self.avg:.02f})"


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
