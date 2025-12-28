import torch
from transformers import Trainer
import torch.nn as nn
from transformers.utils import logging
try:
    from transformers.training_args import OptimizerNames  # newer path
except Exception:
    try:
        from transformers.trainer_utils import OptimizerNames  # some mid versions
    except Exception:
        # Fallback shim for code paths that only check for LOMO/ADALOMO
        print("Warning: OptimizerNames unavailable; using shim with LOMO/ADALOMO only.")
        class OptimizerNames:
            LOMO = "lomo"
            ADALOMO = "adalomo"
from typing import Any, Optional, Union
import torch.distributed as dist
# --- Version-agnostic imports for HF/Accelerate ---

import importlib

# logging is stable
from transformers.utils import logging

# is_sagemaker_mp_enabled moved; prefer transformers.utils, fallback to older path
try:
    from transformers.utils import is_sagemaker_mp_enabled
except Exception:
    try:
        from transformers.trainer_utils import is_sagemaker_mp_enabled  # older
    except Exception:
        print("Warning: is_sagemaker_mp_enabled unavailable; using safe default False.")
        def is_sagemaker_mp_enabled() -> bool:
            return False  # safe default if not available

# smp_forward_backward lives in trainer_pt_utils; guard it (only used on SageMaker MP)
try:
    from transformers.trainer_pt_utils import smp_forward_backward
except Exception:
    def smp_forward_backward(*args, **kwargs):
        raise RuntimeError("smp_forward_backward unavailable: install a Transformers version with SageMaker MP support.")

# Device feature flags: prefer transformers.utils (newer), else integrations (older)
try:
    from transformers.utils import (
        is_torch_xpu_available,
        is_torch_mlu_available,
        is_torch_musa_available,
        is_torch_npu_available,
        is_torch_mps_available,
        is_torch_hpu_available,
    )
except Exception:
    from transformers.integrations import (  # type: ignore
        is_torch_xpu_available,
        is_torch_mlu_available,
        is_torch_musa_available,
        is_torch_npu_available,
        is_torch_mps_available,
        is_torch_hpu_available,
    )

# DistributedType now comes from accelerate; prefer that
try:
    from accelerate.utils import DistributedType
except Exception:
    # very old Transformers re-exported it; try that
    try:
        from transformers.trainer_utils import DistributedType  # type: ignore
    except Exception:
        # Last-resort shim to avoid crashes; you can compare strings instead.
        class DistributedType:
            NO = "NO"
            DATA_PARALLEL = "DATA_PARALLEL"
            DEEPSPEED = "DEEPSPEED"
            TPU = "TPU"
            FSDP = "FSDP"

from updated_transformer.pruning_indices import update_dropout_masks
from updated_transformer.calculate_scores_bert import calculate_scores
import math
def _ddp_is_on():
    return dist.is_available() and dist.is_initialized()

@torch.no_grad()
def _ddp_avg_scores_(scores: dict):
    """
    In-place all-reduce (mean) of every tensor value in `scores`.
    Assumes same keys and shapes on all ranks.
    """
    if not _ddp_is_on():
        return scores
    ws = dist.get_world_size()
    for k, v in scores.items():
        dist.all_reduce(v, op=dist.ReduceOp.SUM)
        v.div_(ws)
    return scores
logger = logging.get_logger(__name__)
# --- helpers (drop next to your Trainer subclass) ---

def _non_special_pool(tokenizer):
    specials = set(getattr(tokenizer, "all_special_ids", []) or [])
    pad_id = getattr(tokenizer, "pad_token_id", None)
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if pad_id is not None: specials.add(pad_id)
    if mask_id is not None: specials.add(mask_id)
    pool = [i for i in range(tokenizer.vocab_size) if i not in specials]
    return pool if pool else list(range(tokenizer.vocab_size))

def make_baseline_ids(target_ids: torch.Tensor,
                      masked_ids: torch.Tensor,
                      tokenizer,
                      kind: str) -> torch.Tensor:
    """
    target_ids: unmasked [B, T]
    masked_ids: already-masked [B, T]
    kind in {'zeroes','random','pad','mask','masked'}
    """
    B, T = target_ids.shape
    dev = target_ids.device
    k = kind.lower()

    if k in ("zeroes", "zeros", "zero"):
        base = torch.zeros_like(target_ids)
    elif k in ("random", "noise", "random_noise"):
        pool = torch.tensor(_non_special_pool(tokenizer), device=dev, dtype=torch.long)
        idx = torch.randint(0, pool.numel(), (B, T), device=dev)
        base = pool[idx]
    elif k.startswith("pad"):
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        base = torch.full_like(target_ids, pad_id)
    elif k.startswith("mask"):
        mask_id = getattr(tokenizer, "mask_token_id", None)
        if mask_id is None:
            print(f"WARNING: tokenizer {tokenizer.__class__.__name__} has no mask token, using pad token as baseline")
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            base = torch.full_like(target_ids, pad_id)
        else:
            base = torch.full_like(target_ids, mask_id)
    elif k in ("masked", "masked_input"):
        base = masked_ids.to(dev)
    else:
        raise ValueError(f"Unknown baseline kind: {kind}")
    return base

def build_conductance_call(inputs: dict, tokenizer, baseline_kind: str, sample_size: int):
    """
    Returns: batches_iterable, baseline_tensor
    - batches_iterable: [(x, y_batch)] where x is UNMASKED ids [b, T]
    - baseline_tensor:  same shape as x, per your calculate_scores(...)
    """
    # choose a small subset along batch dimension
    B = inputs["input_ids"].size(0)
    b = min(sample_size, B)
    idx = torch.arange(b, device=inputs["input_ids"].device)

    x_unmasked = inputs["input_ids_raw"][idx]       # final target for interpolation
    x_masked   = inputs["input_ids"][idx]           # needed for baseline='masked'
    # y_batch is only used by your "Conductance_alt" path; safe default to zeros
    if "labels" in inputs and inputs["labels"].dim() == 1:
        y_batch = inputs["labels"][idx]
    else:
        y_batch = torch.zeros(b, dtype=torch.long, device=x_unmasked.device)

    baseline = make_baseline_ids(x_unmasked, x_masked, tokenizer, baseline_kind)

    # your calculate_scores expects Iterable of (x, y_batch)
    batches = [(x_unmasked, y_batch)]
    return batches, baseline

class HookedTrainer(Trainer):
    def __init__(self, *args, pre_step_hook=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ydrop = False
        self.n_steps = 4
        self.update_samples = 4
        self.update_freq = 1
        self.mask_type = 'rank_loss'
        self.scoring_type = 'Conductance'
        self.after_norm = True
        self.mode = 'mean'
        self.baseline = 'zeroes'
        self.annealing_factor = 0.1
        self.update_batches = 1
        self._micro_in_cycle = 0

        


    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        step = self.state.global_step
        gas = max(1, self.args.gradient_accumulation_steps)
        is_cycle_start = (self._micro_in_cycle == 0) 
        self._micro_in_cycle = (self._micro_in_cycle + 1) % gas

        #print("step:", step)
        
        # update_mask = self.ydrop and step > self.annealing_factor * total_updates and (step + 1) % self.update_freq == 0:
        total_updates = getattr(self.state, "max_steps", None)
        if total_updates is None:
             total_updates = self.state.num_train_epochs * math.ceil(len(self.get_train_dataloader()) / self.args.gradient_accumulation_steps)
        #print("total_updates:", total_updates)

        update_mask = (
            self.ydrop
            and step > self.annealing_factor * total_updates
            and is_cycle_start
            #and (step + 1) % self.update_freq == 0
        )
        if update_mask:
            for drop in model.drop_list:
                drop.use_ydrop()
        #print("ydrop:", self.ydrop, "annealing_factor:", self.annealing_factor, "update_mask:", update_mask)

        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        # Context manager is no-op if CP isn't enabled
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)

            # print("Input keys, types:")
            # for key, value in inputs.items():
            #     print(f"  {key}: {type(value)}")
            batch_count = 0
            if update_mask:
                B = inputs["input_ids"].size(0)
                chunk = max(1, min(self.update_samples, B)) 
                batches = []
                for start in range(0, B, chunk):
                    batch_count += 1
                    end = min(B, start + chunk)
                    sl = slice(start, end)

                    sub_inputs = {
                        "input_ids": inputs["input_ids"][sl],
                    }
                    if "attention_mask" in inputs and isinstance(inputs["attention_mask"], torch.Tensor):
                        sub_inputs["attention_mask"] = inputs["attention_mask"][sl]
                    if "token_type_ids" in inputs and isinstance(inputs["token_type_ids"], torch.Tensor):
                        sub_inputs["token_type_ids"] = inputs["token_type_ids"][sl]

                    labels_sb = inputs.get("labels", None)
                    if labels_sb is not None:
                        labels_sb = labels_sb[sl]

                    # calculate_scores expects (inputs_dict, labels)
                    batches.append((sub_inputs, labels_sb))
                    if batch_count >= self.args.gradient_accumulation_steps:
                        break        
                selected_layers = model.module.selected_layers if hasattr(model, 'module') else model.selected_layers
                scores,_ = calculate_scores(
                        model.module if hasattr(model, 'module') else model,
                        batches, self.args.device, self.scoring_type, mode=self.mode,
                        normalization=False, sm=False, selected_layers=selected_layers, baseline=self.baseline,n_steps = self.n_steps)
                _ddp_avg_scores_(scores)
                # print("Scores shape and values:", {k: v.shape for k, v in scores.items()})
                # print("Scores values type:", {k: type(v) for k, v in scores.items()})
                # print("Scores mean values:", {k: v.mean().item() for k, v in scores.items()})
                # print("Scores std values:", {k: v.std().item() for k, v in scores.items()})
                # print("Scores max values:", {k: v.max().item() for k, v in scores.items()})
                # print("Scores min values:", {k: v.min().item() for k, v in scores.items()})

                update_dropout_masks(
                            model = model.module if hasattr(model, 'module') else model,
                            scores = scores, drop_list=model.module.drop_list if hasattr(model, 'module') else model.drop_list,
                            min_dropout=0.0, noisy_dropout=False,
                            stats=False, alt_attention_cond=False,
                        )
                # for drop in model.drop_list:
                #     print("Set Keep rate vs mean previous vs mean scaling :", drop.base_keep, drop.previous.mean().item(), drop.scaling.mean().item())
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                if is_torch_xpu_available():
                    torch.xpu.empty_cache()
                elif is_torch_mlu_available():
                    torch.mlu.empty_cache()
                elif is_torch_musa_available():
                    torch.musa.empty_cache()
                elif is_torch_npu_available():
                    torch.npu.empty_cache()
                elif is_torch_mps_available():
                    torch.mps.empty_cache()
                elif is_torch_hpu_available():
                    logger.warning(
                        "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                    )
                else:
                    torch.cuda.empty_cache()

            kwargs = {}

            # For LOMO optimizers you need to explicitly use the learning rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                from apex import amp

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
                if (
                    not self.model_accepts_loss_kwargs or num_items_in_batch is None
                ) and self.compute_loss_func is None:
                    # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient accumulation steps
                    loss = loss / self.current_gradient_accumulation_steps

                # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
                # https://github.com/huggingface/transformers/pull/35808
                if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                    kwargs["scale_wrt_gas"] = False

                self.accelerator.backward(loss, **kwargs)

            return loss.detach()
    # def _inner_training_loop(
    #         self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    #     ):
    #     self.accelerator.free_memory()
    #     self._train_batch_size = batch_size
    #     if self.args.auto_find_batch_size:
    #         if self.state.train_batch_size != self._train_batch_size:
    #             from accelerate.utils import release_memory

    #             (self.model_wrapped,) = release_memory(self.model_wrapped)
    #             self.model_wrapped = self.model

    #             # Check for DeepSpeed *after* the initial pass and modify the config
    #             if self.is_deepspeed_enabled:
    #                 # Temporarily unset `self.args.train_batch_size`
    #                 original_bs = self.args.per_device_train_batch_size
    #                 self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
    #                 self.propagate_args_to_deepspeed(True)
    #                 self.args.per_device_train_batch_size = original_bs
    #         self.state.train_batch_size = self._train_batch_size
    #     logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
    #     # Data loader and number of training steps
    #     train_dataloader = self.get_train_dataloader()
    #     if self.is_fsdp_xla_v2_enabled:
    #         train_dataloader = tpu_spmd_dataloader(train_dataloader)

    #     # Setting up training control variables:
    #     # number of training epochs: num_train_epochs
    #     # number of training steps per epoch: num_update_steps_per_epoch
    #     # total number of training steps to execute: max_steps
    #     total_train_batch_size = self.get_total_train_batch_size(args)

    #     (
    #         num_train_epochs,
    #         num_update_steps_per_epoch,
    #         num_examples,
    #         num_train_samples,
    #         epoch_based,
    #         len_dataloader,
    #         max_steps,
    #     ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

    #     num_train_tokens = None
    #     if self.args.include_tokens_per_second:
    #         num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
    #         # If going by epochs, multiply tokens linearly
    #         if len_dataloader is not None and epoch_based:
    #             num_train_tokens *= args.num_train_epochs
    #         # Otherwise since its steps, we just multiply by grad accum
    #         else:
    #             num_train_tokens *= args.gradient_accumulation_steps

    #     if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
    #         if self.args.n_gpu > 1:
    #             # nn.DataParallel(model) replicates the model, creating new variables and module
    #             # references registered here no longer work on other gpus, breaking the module
    #             raise ValueError(
    #                 "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
    #                 " (torchrun or torch.distributed.launch (deprecated))."
    #             )
    #         else:
    #             debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

    #     delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

    #     # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
    #     is_fsdp2 = self.is_fsdp_enabled and (getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2)
    #     if is_fsdp2:
    #         delay_optimizer_creation = False

    #     # We need to reset the scheduler, as its parameters may be different on subsequent calls
    #     if self._created_lr_scheduler:
    #         self.lr_scheduler = None
    #         self._created_lr_scheduler = False

    #     if self.is_deepspeed_enabled:
    #         self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

    #     if not delay_optimizer_creation:
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    #     self.state = TrainerState(
    #         stateful_callbacks=[
    #             cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
    #         ]
    #     )
    #     self.state.is_hyper_param_search = trial is not None
    #     self.state.train_batch_size = self._train_batch_size

    #     # Compute absolute values for logging, eval, and save if given as ratio
    #     self.state.compute_steps(args, max_steps)

    #     # Activate gradient checkpointing if needed
    #     if args.gradient_checkpointing:
    #         self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

    #     model = self._wrap_model(self.model_wrapped)

    #     # as the model is wrapped, don't use `accelerator.prepare`
    #     # this is for unhandled cases such as
    #     # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
    #     use_accelerator_prepare = model is self.model

    #     if use_accelerator_prepare and self.is_fsdp_enabled:
    #         # In case of auto_find_batch_size=True
    #         # Remove FSDP wrapping from sub-models.
    #         self.model = unwrap_model(self.model, recursive=True)

    #     if delay_optimizer_creation:
    #         if use_accelerator_prepare:
    #             # configure fsdp plugin for qlora if any
    #             self._fsdp_qlora_plugin_updates()
    #             if self.accelerator.mixed_precision != "fp8":
    #                 self.model = self.accelerator.prepare(self.model)
    #         self.create_optimizer_and_scheduler(num_training_steps=max_steps)

    #     # prepare using `accelerator` prepare
    #     if use_accelerator_prepare:
    #         self.model.train()
    #         if hasattr(self.lr_scheduler, "step"):
    #             if self.use_apex:
    #                 model = self.accelerator.prepare(self.model)
    #             else:
    #                 # We should avoid accelerate preparing the model in TP case since we dont need it as it is handled by transformers from_pretrained and also it goes into DDP based preparation.
    #                 if self.is_tp_enabled:
    #                     self.optimizer = self.accelerator.prepare(self.optimizer)
    #                 else:
    #                     model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
    #         else:
    #             # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
    #             model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
    #                 self.model, self.optimizer, self.lr_scheduler
    #             )
    #     elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
    #         # In this case we are in DDP + LOMO, which should be supported
    #         self.optimizer = self.accelerator.prepare(self.optimizer)

    #     if self.is_fsdp_enabled:
    #         self.model = self.model_wrapped = model

    #     # for the rest of this function `model` is the outside model, whether it was wrapped or not
    #     if model is not self.model:
    #         self.model_wrapped = model

    #     # backward compatibility
    #     if self.is_deepspeed_enabled:
    #         self.deepspeed = self.model_wrapped

    #     # ckpt loading
    #     if resume_from_checkpoint is not None:
    #         if self.is_deepspeed_enabled:
    #             deepspeed_load_checkpoint(
    #                 self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
    #             )
    #         elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
    #             self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

    #     # Check if saved optimizer or scheduler states exist
    #     self._load_optimizer_and_scheduler(resume_from_checkpoint)
    #     self._load_scaler(resume_from_checkpoint)

    #     # important: at this point:
    #     # self.model         is the Transformers Model
    #     # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
    #     # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

    #     # Train!
    #     logger.info("***** Running training *****")
    #     logger.info(f"  Num examples = {num_examples:,}")
    #     logger.info(f"  Num Epochs = {num_train_epochs:,}")
    #     logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
    #     if self.args.per_device_train_batch_size != self._train_batch_size:
    #         logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
    #     logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
    #     logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    #     logger.info(f"  Total optimization steps = {max_steps:,}")
    #     logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

    #     self.state.epoch = 0
    #     start_time = time.time()
    #     epochs_trained = 0
    #     steps_trained_in_current_epoch = 0
    #     steps_trained_progress_bar = None

    #     # Check if continuing training from a checkpoint
    #     if resume_from_checkpoint is not None and os.path.isfile(
    #         os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
    #     ):
    #         self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
    #         self.compare_trainer_and_checkpoint_args(self.args, self.state)
    #         self._load_callback_state()
    #         epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
    #         if not args.ignore_data_skip:
    #             steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
    #             steps_trained_in_current_epoch *= args.gradient_accumulation_steps
    #         else:
    #             steps_trained_in_current_epoch = 0

    #         logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #         logger.info(f"  Continuing training from epoch {epochs_trained}")
    #         logger.info(f"  Continuing training from global step {self.state.global_step}")
    #         if not args.ignore_data_skip:
    #             logger.info(
    #                 f"  Will skip the first {epochs_trained} epochs then the first"
    #                 f" {steps_trained_in_current_epoch} batches in the first epoch."
    #             )

    #     # Update the references
    #     for attr in ("model", "optimizer", "lr_scheduler"):
    #         setattr(self.callback_handler, attr, getattr(self, attr))
    #     self.callback_handler.train_dataloader = train_dataloader

    #     self.state.init_training_references(self, max_steps, num_train_epochs, trial)

    #     # tr_loss is a tensor to avoid synchronization of TPUs through .item()
    #     tr_loss = torch.tensor(0.0, device=args.device)
    #     # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
    #     self._total_loss_scalar = 0.0
    #     self._globalstep_last_logged = self.state.global_step
    #     model.zero_grad()
    #     grad_norm: Optional[float] = None
    #     learning_rate = None
    #     self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

    #     if args.eval_on_start:
    #         self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

    #     for epoch in range(epochs_trained, num_train_epochs):
    #         epoch_dataloader = train_dataloader
    #         if hasattr(epoch_dataloader, "set_epoch"):
    #             epoch_dataloader.set_epoch(epoch)

    #         # Reset the past mems state at the beginning of each epoch if necessary.
    #         if args.past_index >= 0:
    #             self._past = None

    #         steps_in_epoch = (
    #             len(epoch_dataloader)
    #             if len_dataloader is not None
    #             else args.max_steps * args.gradient_accumulation_steps
    #         )
    #         self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

    #         if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
    #             self._load_rng_state(resume_from_checkpoint)

    #         rng_to_sync = False
    #         steps_skipped = 0
    #         if steps_trained_in_current_epoch > 0:
    #             epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
    #             steps_skipped = steps_trained_in_current_epoch
    #             steps_trained_in_current_epoch = 0
    #             rng_to_sync = True

    #         step = -1
    #         epoch_iterator = iter(epoch_dataloader)
    #         # We chunkify the epoch iterator into gradient accumulation steps `n` batches
    #         remainder = steps_in_epoch % args.gradient_accumulation_steps
    #         if remainder == 0:
    #             remainder = args.gradient_accumulation_steps
    #         update_step = -1
    #         total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
    #             remainder < args.gradient_accumulation_steps
    #         )
    #         for _ in range(total_updates):
    #             update_mask = False
    #             update_step += 1
    #             num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
    #             batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
    #             if self.ydrop and update_step > self.annealing_factor * total_updates and (update_step + 1) % self.update_freq == 0:
    #                 update_mask = True
    #             # Store the number of batches for current gradient accumulation
    #             # This is used to correctly scale the loss when the last accumulation step has fewer batches
    #             self.current_gradient_accumulation_steps = len(batch_samples)
    #             for i, inputs in enumerate(batch_samples):
    #                 step += 1
    #                 do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
    #                 # Since we perform prefetching, we need to manually set sync_gradients
    #                 self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

    #                 if self.args.include_num_input_tokens_seen:
    #                     main_input_name = getattr(self.model, "main_input_name", "input_ids")
    #                     if main_input_name not in inputs:
    #                         logger.warning(
    #                             "Tried to track the number of tokens seen, however the current model is "
    #                             "not configured properly to know what item is the input. To fix this, add "
    #                             "a `main_input_name` attribute to the model class you are using."
    #                         )
    #                     else:
    #                         input_tokens = inputs[main_input_name].numel()
    #                         input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
    #                         self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).sum().item()
    #                 if rng_to_sync:
    #                     self._load_rng_state(resume_from_checkpoint)
    #                     rng_to_sync = False

    #                 # Skip past any already trained steps if resuming training
    #                 if steps_trained_in_current_epoch > 0:
    #                     steps_trained_in_current_epoch -= 1
    #                     if steps_trained_progress_bar is not None:
    #                         steps_trained_progress_bar.update(1)
    #                     if steps_trained_in_current_epoch == 0:
    #                         self._load_rng_state(resume_from_checkpoint)
    #                     continue
    #                 elif steps_trained_progress_bar is not None:
    #                     steps_trained_progress_bar.close()
    #                     steps_trained_progress_bar = None

    #                 if step % args.gradient_accumulation_steps == 0:
    #                     self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

    #                 # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
    #                 context = (
    #                     functools.partial(self.accelerator.no_sync, model=model)
    #                     if i != len(batch_samples) - 1
    #                     and self.accelerator.distributed_type != DistributedType.DEEPSPEED
    #                     else contextlib.nullcontext
    #                 )
    #                 with context():
    #                     tr_loss_step = self.training_step(model, inputs, num_items_in_batch, update_mask=update_mask)

    #                 if (
    #                     args.logging_nan_inf_filter
    #                     and not is_torch_xla_available()
    #                     and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
    #                 ):
    #                     # if loss is nan or inf simply add the average of previous logged losses
    #                     tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
    #                 else:
    #                     if tr_loss.device != tr_loss_step.device:
    #                         raise ValueError(
    #                             f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
    #                         )
    #                     tr_loss = tr_loss + tr_loss_step

    #                 self.current_flos += float(self.floating_point_ops(inputs))

    #                 if do_sync_step:
    #                     # Since we perform prefetching, we need to manually set sync_gradients to True
    #                     self.accelerator.gradient_state._set_sync_gradients(True)

    #                     # Gradient clipping
    #                     if args.max_grad_norm is not None and args.max_grad_norm > 0:
    #                         if is_sagemaker_mp_enabled() and args.fp16:
    #                             _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
    #                         elif self.use_apex:
    #                             from apex import amp

    #                             # Revert to normal clipping otherwise, handling Apex or full precision
    #                             _grad_norm = nn.utils.clip_grad_norm_(
    #                                 amp.master_params(self.optimizer),
    #                                 args.max_grad_norm,
    #                             )
    #                         else:
    #                             grad_norm_context = contextlib.nullcontext
    #                             if self.is_tp_enabled:
    #                                 from torch.distributed._tensor.experimental import implicit_replication

    #                                 grad_norm_context = implicit_replication
    #                             with grad_norm_context():
    #                                 _grad_norm = self.accelerator.clip_grad_norm_(
    #                                     model.parameters(),
    #                                     args.max_grad_norm,
    #                                 )

    #                         if (
    #                             is_accelerate_available()
    #                             and self.accelerator.distributed_type == DistributedType.DEEPSPEED
    #                         ):
    #                             grad_norm = model.get_global_grad_norm()
    #                             # In some cases the grad norm may not return a float
    #                             if hasattr(grad_norm, "item"):
    #                                 grad_norm = grad_norm.item()
    #                         else:
    #                             grad_norm = _grad_norm

    #                     self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

    #                     context = contextlib.nullcontext
    #                     if self.is_tp_enabled:
    #                         from torch.distributed._tensor.experimental import implicit_replication

    #                         context = implicit_replication

    #                     with context():
    #                         self.optimizer.step()

    #                     self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

    #                     # get leaning rate before update
    #                     learning_rate = self._get_learning_rate()

    #                     if not self.accelerator.optimizer_step_was_skipped:
    #                         # Delay optimizer scheduling until metrics are generated
    #                         if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
    #                             self.lr_scheduler.step()

    #                     model.zero_grad()
    #                     self.state.global_step += 1
    #                     self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
    #                     self.control = self.callback_handler.on_step_end(args, self.state, self.control)
    #                     self._maybe_log_save_evaluate(
    #                         tr_loss,
    #                         grad_norm,
    #                         model,
    #                         trial,
    #                         epoch,
    #                         ignore_keys_for_eval,
    #                         start_time,
    #                         learning_rate=learning_rate,
    #                     )
    #                 else:
    #                     self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

    #                 # PyTorch/XLA relies on the data loader to insert the mark_step for
    #                 # each step. Since we are breaking the loop early, we need to manually
    #                 # insert the mark_step here.
    #                 if self.control.should_epoch_stop or self.control.should_training_stop:
    #                     if is_torch_xla_available():
    #                         xm.mark_step()
    #                     break
    #             # We also need to break out of the nested loop
    #             if self.control.should_epoch_stop or self.control.should_training_stop:
    #                 if is_torch_xla_available():
    #                     xm.mark_step()
    #                 break
    #         if step < 0:
    #             logger.warning(
    #                 "There seems not to be a single sample in your epoch_iterator, stopping training at step"
    #                 f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
    #                 f" num_steps ({max_steps}) higher than the number of available samples."
    #             )
    #             self.control.should_training_stop = True

    #         self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
    #         self._maybe_log_save_evaluate(
    #             tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate
    #         )

    #         if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
    #             if is_torch_xla_available():
    #                 # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
    #                 xm.master_print(met.metrics_report())
    #             else:
    #                 logger.warning(
    #                     "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
    #                     "configured. Check your training configuration if this is unexpected."
    #                 )
    #         if self.control.should_training_stop:
    #             break

    #     if args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of training
    #         delattr(self, "_past")

    #     logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    #     if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
    #         # Wait for everyone to get here so we are sure the model has been saved by process 0.
    #         if is_torch_xla_available():
    #             xm.rendezvous("load_best_model_at_end")
    #         elif args.parallel_mode == ParallelMode.DISTRIBUTED:
    #             dist.barrier()
    #         elif is_sagemaker_mp_enabled():
    #             smp.barrier()

    #         self._load_best_model()

    #     # add remaining tr_loss
    #     self._total_loss_scalar += tr_loss.item()
    #     effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
    #     train_loss = self._total_loss_scalar / effective_global_step

    #     metrics = speed_metrics(
    #         "train",
    #         start_time,
    #         num_samples=num_train_samples,
    #         num_steps=self.state.max_steps,
    #         num_tokens=num_train_tokens,
    #     )
    #     self.store_flos()
    #     metrics["total_flos"] = self.state.total_flos
    #     metrics["train_loss"] = train_loss

    #     self.is_in_train = False

    #     self._memory_tracker.stop_and_update_metrics(metrics)

    #     self.log(metrics)

    #     run_dir = self._get_output_dir(trial)
    #     checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

    #     # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
    #     if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
    #         for checkpoint in checkpoints_sorted:
    #             if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
    #                 logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
    #                 shutil.rmtree(checkpoint, ignore_errors=True)

    #     self.control = self.callback_handler.on_train_end(args, self.state, self.control)

    #     # Wait for the checkpoint to be uploaded.
    #     self._finish_current_push()

    #     # After training we make sure to retrieve back the original forward pass method
    #     # for the embedding layer by removing the forward post hook.
    #     if self.neftune_noise_alpha is not None:
    #         self._deactivate_neftune(self.model)

    #     return TrainOutput(self.state.global_step, train_loss, metrics)