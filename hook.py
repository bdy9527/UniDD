import os
import numpy as np
import torch
from torch.linalg import solve


class BNFeatureHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().reshape([nch, -1]).var(1, unbiased=False)
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)
        self.r_feature = r_feature

    def close(self):
        self.hook.remove()


class ConvFeatureHook:
    def __init__(self, module, name, save_path, dataset,
                 filter='cov', signal='mean', feat='input', ema=True,
                 device='cuda:0'):

        self.module = module
        self.name = name
        self.dataset = dataset
        self.device = device

        self.filter = filter
        self.signal = signal
        self.feat = feat
        self.ema = ema

        self.beta = 0.
        self.scheduler = None

        if module is not None and name is not None:
            self.hook = module.register_forward_hook(self.post_hook_fn)
        else:
            raise ModuleNotFoundError("module and name can not be None!")

        # For Tiny-ImageNet
        if self.dataset == 'Tiny':
            self.num_classes = 200
            self.data_number = 100000
        elif self.dataset == 'ImageNet':
            self.num_classes = 1000
            self.data_number = 1281167
        elif self.dataset == 'CIFAR-100':
            self.num_classes = 100
            self.data_number = 50000
        elif self.dataset == 'CIFAR-10':
            self.num_classes = 10
            self.data_number = 50000
        else:
            return

        # Caching statistics
        self.batch_id = 0.
        self.onehot_labels = 0.
        self.pseudo_labels = 0.
        self.num_per_classes = 0.

        self.cov = 0.
        self.corr = 0.
        self.mean = 0.
        self.mix_mean = 0.
        self.class_mean = 0.

        self.batch_cov = 0.
        self.batch_corr = 0.
        self.batch_mean = 0.
        self.batch_mix_mean = 0.
        self.batch_class_mean = 0.

        if self.feat == 'input':
            save_path = os.path.join(save_path, self.dataset, 'input')
        elif self.feat == 'output':
            save_path = os.path.join(save_path, self.dataset, 'output')
        else:
            print('error feat', self.feat)
            return

        dir = os.path.join(save_path, "ConvFeatureHook", name)
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

        self.save_path = os.path.join(save_path, "ConvFeatureHook", name, "running.npz")
        print(self.save_path)

        if os.path.exists(self.save_path):
            npz_file = np.load(self.save_path)
            self.load_tag = True
            self.running_cov  = torch.from_numpy(npz_file["running_cov"]).to(device)
            self.running_corr = torch.from_numpy(npz_file["running_corr"]).to(device)
            self.running_mean = torch.from_numpy(npz_file["running_mean"]).to(device)
            self.running_mix_mean = torch.from_numpy(npz_file["running_mix_mean"]).to(device)
            self.running_class_mean = torch.from_numpy(npz_file["running_class_mean"]).to(device)
            print(self.running_cov.shape)

        else:
            self.load_tag = False
            self.running_cov  = 0.
            self.running_corr = 0.
            self.running_mean = 0.
            self.running_mix_mean = 0.
            self.running_class_mean = 0.

    def save(self):
        npz_file = {"running_cov": self.running_cov.cpu().numpy() 
                    if isinstance(self.running_cov, torch.Tensor) else self.running_cov,

                    "running_corr": self.running_corr.cpu().numpy() 
                    if isinstance(self.running_corr, torch.Tensor) else self.running_corr,

                    "running_mean": self.running_mean.cpu().numpy() 
                    if isinstance(self.running_mean, torch.Tensor) else self.running_mean,

                    "running_mix_mean": self.running_mix_mean.cpu().numpy()
                    if isinstance(self.running_mix_mean, torch.Tensor) else self.running_mix_mean,

                    "running_class_mean": self.running_class_mean.cpu().numpy()
                    if isinstance(self.running_class_mean, torch.Tensor) else self.running_class_mean}
        
        np.savez(self.save_path, **npz_file)

    def set_hook(self, pre=True):
        if hasattr(self, "hook"):
            self.close()
        if pre:
            self.hook = self.module.register_forward_hook(self.pre_hook_fn)
        else:
            self.hook = self.module.register_forward_hook(self.post_hook_fn)

    @torch.no_grad()
    def pre_hook_fn(self, module, input, output):
        if self.feat == 'input':
            bs, nch, h, w = input[0].shape
            feat = input[0]
        elif self.feat == 'output':
            bs, nch, h, w = output.shape
            feat = output

        x = feat.permute(1, 0, 2, 3).contiguous().reshape(nch, -1)   # [C, B*H*W]
        cov = torch.cov(x, correction=0)
        corr = torch.mm(x, x.T)

        mean = feat.mean([0, 2, 3])

        # [B, K] -> [1, K]
        self.num_per_classes += self.onehot_labels.sum(axis=0, keepdim=True)

        # [B, C, H, W] -> [C, B] * [B, K]
        class_mean = feat.mean([2, 3]).T @ self.onehot_labels
        mix_mean = feat.mean([2, 3]).T @ self.pseudo_labels

        self.running_cov  += (cov * bs / self.data_number)
        self.running_corr += (corr / (h * w) / self.data_number)
        self.running_mean += (mean * bs / self.data_number)
        self.running_mix_mean += (mix_mean / self.data_number)
        self.running_class_mean += (class_mean / self.data_number)

    def post_hook_fn(self, module, input, output):
        if self.feat == 'input':
            bs, nch, h, w = input[0].shape
            feat = input[0]
        elif self.feat == 'output':
            bs, nch, h, w = output.shape
            feat = output

        '''For Filters'''
        x = feat.permute(1, 0, 2, 3).contiguous().reshape(nch, -1)   # [C, B*H*W]

        cov = torch.cov(x, correction=0)
        self.batch_cov = cov.detach()

        if self.ema:
            cov = (1 - 1 / self.batch_id) * self.cov + (1 / self.batch_id) * cov

        if self.filter == 'HFM':
            I = torch.eye(cov.shape[0], device=cov.device)
            filter_t = solve(self.running_cov + self.beta * I, I)
            filter_s = solve(             cov + self.beta * I, I)

            # filter_t = inv(self.running_cov + self.beta * torch.eye(cov.shape[0], device=cov.device))
            # filter_s = inv(             cov + self.beta * torch.eye(cov.shape[0], device=cov.device))

        elif self.filter == 'LFM':
            filter_t = self.running_cov
            filter_s =              cov

        else:
            print('error filter', self.filter)
            return

        # if self.filter == 'cov':
        #     cov = torch.cov(x, correction=0)
        #     self.batch_cov  = cov.detach()

        #     if self.ema:
        #         cov = (1 - 1 / self.batch_id) * self.cov + (1 / self.batch_id) * cov

        #     filter_t = self.running_cov + self.beta * torch.eye(cov.shape[0], device=cov.device)
        #     filter_s = cov + self.beta * torch.eye(cov.shape[0], device=cov.device)

        # elif self.filter == 'corr':
        #     corr = torch.mm(x, x.T) / (bs * h * w)
        #     self.batch_corr = corr.detach()

        #     if self.ema:
        #         corr = (1 - 1 / self.batch_id) * self.corr + (1 / self.batch_id) * corr

        #     filter_t = self.running_corr + self.beta * torch.eye(corr.shape[0], device=corr.device)
        #     filter_s = corr + self.beta * torch.eye(corr.shape[0], device=corr.device)

        # else:
        #     print('error filter', self.filter)
        #     return


        '''For Signals'''
        if self.signal == 'mean':
            mean = feat.mean([0, 2, 3])
            self.batch_mean = mean.detach()

            if self.ema:
                mean = (1 - 1 / self.batch_id) * self.mean + (1 / self.batch_id) * mean

            signal_t = self.running_mean
            signal_s = mean

        elif self.signal == 'class':
            class_mean = (feat.mean([2, 3]).T @ self.onehot_labels) / bs
            self.batch_class_mean = class_mean.detach()

            if self.ema:
                class_mean = (1 - 1 / self.batch_id) * self.class_mean + (1 / self.batch_id) * class_mean 

            signal_t = self.running_class_mean
            signal_s = class_mean

        elif self.signal == 'mix':
            mix_mean = (feat.mean([2, 3]).T @ self.pseudo_labels) / bs
            self.batch_mix_mean = mix_mean.detach()

            if self.ema:
                mix_mean = (1 - 1 / self.batch_id) * self.mix_mean + (1 / self.batch_id) * mix_mean

            signal_t = self.running_mix_mean
            signal_s = mix_mean

        else:
            print('error signal', self.signal)
            return

        filter_loss = torch.norm(filter_t - filter_s, 'fro')
        signal_loss = torch.norm(filter_t @ signal_t - filter_s @ signal_s, 2)

        self.r_feature = filter_loss + signal_loss

    @torch.no_grad()
    def ema_update(self):

        self.cov = (1 - 1 / self.batch_id) * self.cov + (1 / self.batch_id) * self.batch_cov
        self.cov = self.cov.detach()

        # if self.filter == 'cov':
        #     self.cov = (1 - 1 / self.batch_id) * self.cov + (1 / self.batch_id) * self.batch_cov
        #     self.cov = self.cov.detach()
        # elif self.filter == 'corr':
        #     self.corr = (1 - 1 / self.batch_id) * self.corr + (1 / self.batch_id) * self.batch_corr
        #     self.corr = self.corr.detach()
        # else:
        #     print('error filter', self.filter)
        #     return

        if self.signal == 'mean':
            self.mean = (1 - 1 / self.batch_id) * self.mean + (1 / self.batch_id) * self.batch_mean
            self.mean = self.mean.detach()
        elif self.signal == 'class':
            self.class_mean = (1 - 1 / self.batch_id) * self.class_mean + (1 / self.batch_id) * self.batch_class_mean        
            self.class_mean = self.class_mean.detach()
        elif self.signal == 'mix':
            self.mix_mean = (1 - 1 / self.batch_id) * self.mix_mean + (1 / self.batch_id) * self.batch_mix_mean        
            self.mix_mean = self.mix_mean.detach()
        else:
            print('error signal', self.signal)
            return

    def close(self):
        self.hook.remove()
