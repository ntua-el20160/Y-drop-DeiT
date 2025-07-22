
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import random
from pathlib import Path
from updated_transformer.plots import plot_epoch_statistics

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset, create_subdataset
from engine import train_one_epoch, evaluate
#from losses import DistillationLosss
from samplers import RASampler
#from augment import new_data_aug_generator
import models
import utils
from evaluate_gradients.MultiLayerConductance import MultiLayerConductance   
import torch
import torchvision.transforms as T
from torchvision.models import resnet18
from PIL import Image
import matplotlib.pyplot as plt
def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_tiny_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop_rate', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')

    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')


    parser.add_argument('--experiment_name', default='simpletransformer', type=str, help='experiment name')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    # parser.add_argument('--train-mode', action='store_true')
    # parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    # parser.set_defaults(train_mode=True)


    #parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    # parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

# # Distillation parameters
#     parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
#                         help='Name of teacher model to train (default: "regnety_160"')
#     parser.add_argument('--teacher-path', type=str, default='')
#     parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
#     parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
#     parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
#     # * Cosub params
#     parser.add_argument('--cosub', action='store_true') 
    
#     # * Finetuning params
#     parser.add_argument('--finetune', default='', help='finetune from checkpoint')
#     parser.add_argument('--attn-only', action='store_true') 
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01_101/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR10','CIFAR100', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    #parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    #parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument("--dist-eval",
    action="store_true",
    default=False,
    help="Use DistributedSampler for validation",
    )
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # --- Custom Dropout Hyperparameters ---
    parser.add_argument('--ydrop', action='store_true', default=True,
                    help='Enable Y-Drop (MyDropout) by default')
    parser.add_argument('--no-ydrop', dest='ydrop', action='store_false',
                    help='Disable Y-Drop (MyDropout)')

    parser.add_argument('--elasticity', type=float, default=0.01,
                        help='Elasticity factor for custom dropout')

    parser.add_argument('--annealing_factor', type=float, default=5,
                        help='Annealing factor for custom dropout')
    parser.add_argument('--n_steps',type=int,default = 5,
                         help ='intermediate steps for conductance calculation')
    parser.add_argument('--update_batches',type=int,default = 1,
                         help ='intermediate steps for conductance calculation')
    parser.add_argument('--update_freq',type=int,default = 1,
                            help ='intermediate steps for conductance calculation')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Number of epochs with no improvement in eval loss before early stopping')
    parser.add_argument('--plot_freq', default=5, type=int, help='plot frequency')
    parser.add_argument('--scaler', default=1.0, type=float, help='Loss scaler for mixed precision training')
    parser.add_argument('--mask_type', default='sigmoid', type=str, help='Type of mask for dropout')
    
    parser.add_argument('--sub_dataset', default ='none',choices=['none','stratified', 'random'] ,type=str, help='Sub dataset to use for training')
    parser.add_argument('--sub_factor', default=10, type=int, help='Sub dataset factor')
    parser.add_argument('--update_scaling',choices=['no','increasing', 'decreasing'], default='no', type =str,
                        help='Scale update frequency  for custom dropout')
    parser.add_argument('--update_scaling_steps', default=5, type=int, help='Amount of frequency updates')
    parser.add_argument('--scoring-type', choices=['Conductance', 'Sensitivity',"Conductance_alt"], default='Conductance',
                        type=str, help='Scoring type for custom dropout')
    parser.add_argument('--same_batch', action='store_true', default=False,
                        help='Enable smooth scoring for custom dropout')
    parser.add_argument('--transformer_mean', action='store_true', default=False,
                        help='Enable smooth scoring for custom dropout')
    parser.add_argument('--noisy_score', action='store_true', default=False,
                        help='Noise addition to score')
    parser.add_argument('--noisy_dropout', action='store_true', default=False,
                        help='Noise addition to dropout')
    parser.add_argument('--min_dropout', type=float, default=0.0,
                help='Minimum allowed dropout rate')
    parser.add_argument('--after_norm', action='store_true', default=False,
                help='Calculate conductance after normalization')
    
    parser.add_argument('--alt_attention_cond', action='store_true', default=False,
                help='Calculate conductance after normalization')
    parser.add_argument('--rescaling_type',choices=['linear','piecewise', 'power_law'], default=None, type =str,
                    help='Method to rescale the the limits of the dropout masks')
    
    return parser
def main(args):
    utils.init_distributed_mode(args)

    # print(args)
    
    # if args.distillation_type != 'none' and args.finetune and not args.eval:
    #     raise NotImplementedError("Finetuning with distillation not yet supported")
    # device = torch.device(args.device)

    # # fix the seed for reproducibility
    # seed = args.seed + utils.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # # random.seed(seed)

    # cudnn.benchmark = True
    seed = args.seed

    # 1. Python built-in RNG
    random.seed(seed)
    # 2. NumPy RNG
    np.random.seed(seed)
    # 3. Torch CPU RNG
    torch.manual_seed(seed)
    # 4. Torch CUDA RNGs (if you have GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 5. Enforce deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # random.seed(seed)

    #cudnn.benchmark = True
    device = torch.device(args.device)
    from torchvision.datasets import CIFAR10
    raw_ds = CIFAR10(root=args.data_path, train=True, download=False, transform=None)
    img_sample, label_sample = raw_ds[0]


    model = create_model(
        args.model,
        pretrained=False,
        num_classes=10,
        drop_rate=args.drop_rate,   # changed from --drop
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        # pass our extra custom keys. You can add them here:
        ydrop=args.ydrop,
        mask_type=args.mask_type,
        elasticity=args.elasticity,
        scaler=args.scaler,
        n_steps=args.n_steps,
        transformer_mean=args.transformer_mean,
        rescaling_type=args.rescaling_type,
        )
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        ema_device = torch.device('cpu') if args.model_ema_force_cpu else device
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device=ema_device,
            resume='')
    try:
        print(f"Resuming from checkpoint: {args.resume}")

        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        
        model.load_state_dict(checkpoint['model'])

        if args.model_ema:
            utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
        history = checkpoint.get('history', {})
        if history:
            for i, drop in enumerate(model.drop_list):
                drop_history = history.get(f'drop{i}', {})
                drop.progression_keep = drop_history.get('progression_keep', [])
                drop.progression_scoring = drop_history.get('progression_scoring', [])
        
        best_loss = checkpoint.get('lowest_loss', float('inf'))
        #cumulative_train_time = checkpoint.get('train_time', 0.0)
        best_acc = checkpoint.get('best_acc', 0.0)
    except Exception as e:
        print(f"Failed to resume from checkpoint: {e}")
        raise RuntimeError(f"Failed to resume from checkpoint: {e}")
    
           

    for i,block in enumerate(model.blocks):
        model.selected_layers[i*4 + 1] = block.norm2

    for i in range(len(model.blocks)-1):
        model.selected_layers[i*4 + 3] = model.blocks[i+1].norm1
    model.eval()


    # 2) prepare a single test image


    # if your dataset returns a Tensor, convert to PIL:
    if isinstance(img_sample, torch.Tensor):
        img_pil = T.ToPILImage()(img_sample)
    else:
        img_pil = img_sample
    tf = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225])
    ])
    inp = tf(img_pil).unsqueeze(0).requires_grad_()
    
    baseline = torch.zeros_like(inp)

    # 3) compute conductance + IG
    from captum.attr import LayerConductance 
    label_tensor = torch.tensor([label_sample], dtype=torch.long, device=inp.device)

    cond = MultiLayerConductance(model.crit_for,  model.selected_layers)
    layer_attr, input_attr, delta = cond.attribute(
        inp,
        baselines=baseline,
        #target=label_sample,           # e.g. “bull mastiff”
        n_steps=5,
        additional_forward_args=(label_tensor,),
        return_convergence_delta=True,
        return_input_attributions=True,
    )
    # cond = MultiLayerConductance(model,  model.selected_layers)
    # layer_attr, input_attr, delta = cond.attribute(
    #     inp,
    #     baselines=baseline,
    #     target=label_sample,           # e.g. “bull mastiff”
    #     n_steps=5,
    #     return_convergence_delta=True,
    #     return_input_attributions=True,
    #     method="riemann_trapezoid",
    # )
    for i, attr in enumerate(layer_attr):
        if i<5:
            simple_layer_cond = LayerConductance(model, model.selected_layers[i])
            lc_attr, lc_delta = simple_layer_cond.attribute(
                inp, baselines=baseline, n_steps=5,target=label_sample,
                return_convergence_delta=True,method="riemann_trapezoid",
)
            print(f"Layer {i+1} conductance shape:", attr.shape)
            print(f"Layer {i+1} conductance:", attr.sum().item())
            # print(f"Layer {i+1} conductance (normalized):", attr.sum().item() / inp.numel())
            print(f"Layer {i+1} conductance (mean),variance(var):", attr.mean().item(), attr.var().item())
            print("Convergence δ:", delta[i].item())
            print("LC attr:", lc_attr.sum().item(),"LC mean:", lc_attr.mean().item(), "LC var:", lc_attr.var().item()," δ:", lc_delta)
    # print(input_attr[0].shape)
    # print("Image type:",label_sample)
    # 4) plot
    attr_tensor = input_attr[0].squeeze(0).detach()        # [3,224,224], no grad
         # [3,H,W]
    heatmap = attr_tensor.abs().sum(0).cpu().numpy()  # [H,W]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    # 7) convert original to numpy [H,W,3] in [0,1]
    orig = np.array(img_pil).astype(np.float32) / 255.0

    # 8) plot side‑by‑side
    h, w = heatmap.shape

# Option A: simple imshow with Reds colormap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.imshow(orig)
    ax1.set_title("Original")
    ax1.axis("off")

    ax2.imshow(orig)
    # overlay only the red channel:
    ax2.imshow(heatmap, cmap="Reds", alpha=0.6)
    ax2.set_title("IG as Red Map")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

