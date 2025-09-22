from transformers import Trainer
from typing import Optional, Dict, Any
import torch
import os
# ---- stdlib / torch ----
import os, time, shutil, contextlib, functools
import torch
import torch.distributed as dist
import torch.nn as nn
# ---- transformers bits that _inner_training_loop uses ----
from transformers import Trainer
from transformers.trainer_callback import TrainerState, ExportableState
from transformers.trainer_utils import  TrainOutput, speed_metrics
from transformers.training_args import OptimizerNames, ParallelMode
from transformers.trainer_pt_utils import get_model_param_count
from transformers.modeling_utils import unwrap_model
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.utils import (
    is_accelerate_available,
    is_torch_xla_available,
    is_sagemaker_mp_enabled,
)
from transformers.utils import logging as hf_logging
from updated_transformer.dynamic_dropout import MyDropout

from bert_helpers.calculate_scores_text import calculate_mlm_scores
from typing import List, Tuple
def update_dropout_masks(
    model: torch.nn.Module,
    scores: Dict[int, torch.Tensor],
    drop_list = None,
    alt_attention_cond: bool = False,
    stats: bool = False,
    min_dropout: float = 0.0,
    noisy_dropout: bool = False
):
    if drop_list is None:
        drop_list = model.drop_list

    for i, drop_layer in enumerate(drop_list):
        score = scores[i]
            
        drop_layer.update_dropout_masks(score, stats=stats,noisy = noisy_dropout,min_dropout=min_dropout)
def collect_encoder_dropouts_and_selected_layers(model) -> Tuple[List[nn.Dropout], List[nn.Module]]:
    """
    Returns:
        drop_list:    [Dropout modules], in the linear order they appear *within the encoder*.
        selected_layers: [Modules] aligned to the dropouts as follows:
            - For each '...attention.self.dropout'  -> append its .query, then .key
            - For each '...attention.output.dropout'-> append its .LayerNorm (BertSelfOutput.LayerNorm)
            - For each '...output.dropout' (non-attention) -> append its .LayerNorm (BertOutput.LayerNorm)
        Note: Embedding/head dropouts are intentionally excluded so that the mapping
              cleanly matches the attention/FFN structures you requested.
    """
    drop_list: List[nn.Dropout] = []
    selected_layers: List[nn.Module] = []

    name_to_module = dict(model.named_modules())
    encoder_prefix = "bert.encoder"

    for name, module in model.named_modules():
        if not name.startswith(encoder_prefix):
            continue
        if isinstance(module, MyDropout):
            drop_list.append(module)

            # Case 1: Self-attention dropout -> want .query and .key from the same scope
            if name.endswith("attention.self.dropout"):
                base = name.rsplit(".dropout", 1)[0]  # ...attention.self
                q = name_to_module.get(base + ".query", None)
                k = name_to_module.get(base + ".key", None)
                if q is not None:
                    selected_layers.append(q)
                if k is not None:
                    selected_layers.append(k)

            # Case 2: Attention output dropout -> want BertSelfOutput.LayerNorm
            elif name.endswith("attention.output.dropout"):
                base = name.rsplit(".dropout", 1)[0]  # ...attention.output
                ln = name_to_module.get(base + ".LayerNorm", None)
                if ln is not None:
                    selected_layers.append(ln)

            # Case 3: FFN output dropout (but not the attention output one) -> want BertOutput.LayerNorm
            elif name.endswith("output.dropout") and ".attention." not in name:
                base = name.rsplit(".dropout", 1)[0]  # ...layer.X.output
                ln = name_to_module.get(base + ".LayerNorm", None)
                if ln is not None:
                    selected_layers.append(ln)

    return drop_list, selected_layers
# logger equivalent to the one used in Trainer
logger = hf_logging.get_logger(__name__)
try:
    from transformers.trainer_utils import TRAINER_STATE_NAME
except Exception:
    try:
        from transformers.trainer import TRAINER_STATE_NAME
    except Exception:
        TRAINER_STATE_NAME = "trainer_state.json"

# ---- accelerate items used by _inner_training_loop ----
# (Only needed if accelerate is installed; Trainer already guards usage)
try:
    from accelerate import skip_first_batches
    from accelerate.utils import DistributedType
except Exception:  # accelerate not installed or too old
    skip_first_batches = None
    class DistributedType:  # tiny stub to avoid NameError; won't be used without accelerate
        DEEPSPEED = object()

# ---- optional integrations (guarded like Trainer does) ----
# TPU / XLA
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
else:
    xm = None
    met = None

# SageMaker MP (only if available)
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp  # noqa: F401
else:
    smp = None

# PEFT helper used by Trainer
try:
    # private but exported at module level in transformers.trainer
    from transformers.trainer import _is_peft_model
except Exception:
    # fallback: works if peft is installed
    def _is_peft_model(model):
        try:
            from peft import PeftModel
            return isinstance(model, PeftModel)
        except Exception:
            return False

class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        A = self.args  # shorthand
        self.ydrop               = getattr(A, "ydrop", False)
        self.ypath               = getattr(A, "ypath", False)
        self.annealing_factor    = getattr(A, "annealing_factor", 0)
        self.switch_epoch        = getattr(A, "switch_epoch", None)
        self.update_freq         = getattr(A, "update_freq", 1)
        self.scoring_type        = getattr(A, "scoring_type", "Conductance")
        self.mask_type           = getattr(A, "mask_type", "mask")
        self.cond_select_mode    = getattr(A, "cond_select_mode", "")
        self.update_batches      = getattr(A, "update_batches", 0)
        self.conductance_batch_size = getattr(A, "conductance_batch_size", None)
        self.tracker             = getattr(A, "tracker", None)
        self.epoch_gap           = getattr(A, "epoch_gap", 1)
        # NEW: for MLM conductance
        self.baseline            = getattr(A, "baseline", "mask")
        self.target_mode         = getattr(A, "target_mode", "pred")
        self.n_steps             = getattr(A, "n_steps", 5)
        self.mode                = getattr(A, "mode", None)
        self.switch_pct          = getattr(A, "switch_pct", None)
        
        if not hasattr(self, "drop_list") or not hasattr(self, "selected_layers"):
            try:
                self.drop_list, self.selected_layers = collect_encoder_dropouts_and_selected_layers(self.model)
            except Exception as e:
                logger.error(f"Error collecting dropouts and selected layers: {e}")

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the initial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self.get_total_train_batch_size(args)

        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
        is_fsdp2 = self.is_fsdp_enabled and (getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2)
        if is_fsdp2:
            delay_optimizer_creation = False

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = model is self.model

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    # We should avoid accelerate preparing the model in TP case since we dont need it as it is handled by transformers from_pretrained and also it goes into DDP based preparation.
                    if self.is_tp_enabled:
                        self.optimizer = self.accelerator.prepare(self.optimizer)
                    else:
                        model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0, device=args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        learning_rate = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            #########################################
            ###NEW CODE###
            # if self.ydrop:
            #     support_iterator = iter(epoch_dataloader)
            # use_ydrop_now = (
            #     self.ydrop
            #     and epoch >= self.annealing_factor
            #     and (self.switch_epoch is None or epoch < self.switch_epoch)
            # )

            # for d in self.drop_list:
            #     d.use_ydrop() if use_ydrop_now else d.use_normal_dropout()

            save_this_epoch = (self.tracker is not None) and (epoch % self.epoch_gap == 0)
            if save_this_epoch:
                self.tracker.begin_epoch(epoch)
            #########################################    
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = steps_in_epoch % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
                remainder < args.gradient_accumulation_steps
            )
            #########################################
            ###NEW CODE###
            def _progress_ratio(epoch_idx: int, update_idx: int) -> float:
                """
                Fraction of total training completed in [0,1]:
                - Step-based when max_steps>0
                - Else epoch+update-based.
                """
                if not epoch_based and max_steps is not None and max_steps > 0:
                    # self.state.global_step = completed optimizer steps so far
                    return min(1.0, float(self.state.global_step) / float(max_steps))
                # epoch-based fallback
                total_updates_all = max(1, num_train_epochs * total_updates)
                done_updates = epoch_idx * total_updates + max(0, update_idx)
                print(f"Progress: {done_updates}/{total_updates_all}")
                return min(1.0, float(done_updates) / float(total_updates_all))

            def _use_ydrop_now(epoch_idx: int, update_idx: int) -> bool:
                """
                Decides if Y-Drop is active at the current point in training.
                If 0<annealing_factor<=1 => treat as % cutoff (first X% of training).
                If annealing_factor>1    => keep old epoch-based behavior.
                Optional: switch_pct to end Y-Drop by % cutoff.
                """
                print('start')
                if not self.ydrop:
                    return False
                print('prog')
                p = _progress_ratio(epoch_idx, update_idx)
                print(f"Progress ratio: {p:.2f} (epoch {epoch_idx}, update {update_idx})")
                if 0 < float(self.annealing_factor) <= 1.0:
                    # Percent-of-training mode
                    end_pct = float(self.annealing_factor)
                    print(end_pct)
                    in_window = (p > end_pct)
                    if self.switch_pct is not None:
                        in_window = (p < float(self.switch_pct))
                    print(f"Y-Drop active: {in_window}")
                    return in_window

                # Old epoch-based behavior for backward-compat
                start_ep = int(self.annealing_factor) if self.annealing_factor is not None else 0
                in_window = (epoch_idx >= start_ep)
                if self.switch_epoch is not None:
                    in_window = in_window and (epoch_idx < int(self.switch_epoch))
                return in_window
            if self.ydrop:
                support_iterator = iter(epoch_dataloader)
            #########################################
            for up_num in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                # print("Batch samples:")
                # print(type(batch_samples))
                # #print(batch_samples[0].shape)
                # print(len(batch_samples[0]))
                #########################################
                ###NEW CODE###
                
                use_ydrop_now = _use_ydrop_now(epoch, up_num)
                if self.ydrop:
                    for d in self.drop_list:
                        d.use_ydrop() if use_ydrop_now else d.use_normal_dropout()
                
                if (use_ydrop_now or save_this_epoch) and self.update_batches > 0 and (up_num % self.update_freq == 0):
                    support_batch_samples, _ = self.get_batch_samples(support_iterator, num_batches, args.device)
                    print(type(support_batch_samples))

        # Compute conductance on support micro-batches
                    scores, _ = calculate_mlm_scores(
                        model=self.model,                      # use the (wrapped) model in this loop
                        tokenizer=self.tokenizer,
                        batches=support_batch_samples,
                        device=args.device,
                        scoring_type=self.scoring_type,
                        baseline=self.baseline,
                        target_mode=self.target_mode,
                        selected_layers=self.selected_layers,
                        n_steps=self.n_steps,
                        normalization=False,
                        sm=False,
                        mode=self.mode,
                    )
                    for score in scores.values():
                        print(score.shape,score.mean(),score.std())
                    
                    # new_scores = {}
                    # num_layers = len(scores) // 4
                    # A = self.model.config.num_attention_heads
                    # for i in range(num_layers):
                    #     q = scores[4 * i]       # expect shape [H] after your reductions
                    #     k = scores[4 * i + 1]   # expect shape [H]
                    #     ln_attn = scores[4 * i + 2]  # [H]
                    #     ln_ffn  = scores[4 * i + 3]  # [H]
                    #     def to_heads(x,A):
                    #         L, H = x.shape
                    #         Hd = H // A
                    #         return x.view(L, A, Hd).permute(1, 0, 2).contiguous()
                    #     # Outer product to get an [H, H] “importance” matrix for the qk score space

                    #     attn_score_importance = to_heads(q, A) @ to_heads(k, A).transpose(-1, -2)

                    #     # Store aligned to the 3 dropouts of this transformer layer:
                    #     #  - 3*i:    attention.self.dropout   (matrix)
                    #     #  - 3*i+1:  attention.output.dropout (vector)
                    #     #  - 3*i+2:  output.dropout          (vector)
                    #     new_scores[3 * i]     = attn_score_importance
                    #     new_scores[3 * i + 1] = ln_attn
                    #     new_scores[3 * i + 2] = ln_ffn

                    if save_this_epoch:
                        self.tracker.update(scores)

                    if use_ydrop_now:
                        update_dropout_masks(
                            model=self.model,
                            scores=scores,
                            drop_list=self.drop_list,
                            alt_attention_cond=False,
                            stats=False,
                            min_dropout=0.0,
                            noisy_dropout=False,
                        )

                #########################################
                # Store the number of batches for current gradient accumulation
                # This is used to correctly scale the loss when the last accumulation step has fewer batches
                self.current_gradient_accumulation_steps = len(batch_samples)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).sum().item()
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                        else contextlib.nullcontext
                    )
                    with context():
                        tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                from apex import amp

                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                grad_norm_context = contextlib.nullcontext
                                if self.is_tp_enabled:
                                    from torch.distributed._tensor.experimental import implicit_replication

                                    grad_norm_context = implicit_replication
                                with grad_norm_context():
                                    _grad_norm = self.accelerator.clip_grad_norm_(
                                        model.parameters(),
                                        args.max_grad_norm,
                                    )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        context = contextlib.nullcontext
                        if self.is_tp_enabled:
                            from torch.distributed._tensor.experimental import implicit_replication

                            context = implicit_replication

                        with context():
                            self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        # get leaning rate before update
                        learning_rate = self._get_learning_rate()

                        if not self.accelerator.optimizer_step_was_skipped:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(
                            tr_loss,
                            grad_norm,
                            model,
                            trial,
                            epoch,
                            ignore_keys_for_eval,
                            start_time,
                            learning_rate=learning_rate,
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                ############################################
                ###NEW CODE###
                if save_this_epoch:
                    self.tracker.end_epoch()
                ############################################
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(
                tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate
            )

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

