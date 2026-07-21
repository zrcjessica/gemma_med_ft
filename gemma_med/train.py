"""Medical SFT for Gemma 3, sizes 1b--27b.

Recipe notes (see docs/RECIPE.md for the full account of what we can and cannot
match):

* Loss is computed on assistant turns only. This mirrors the one concrete thing
  the MedGemma report says about its fine-tuning objective -- "cross-entropy on
  completion only".
* The learning rates published in the report ({1e-7, 5e-7, 1e-6}) are for
  adapting an already-instruction-tuned MedGemma to a narrow downstream task,
  NOT for this stage. Using them here underfits badly. Our defaults are
  conventional Gemma 3 SFT values; sweep with scripts/sweep_lr.sh.
* Gemma 3 4b+ checkpoints declare the Gemma3ForConditionalGeneration multimodal
  wrapper; 1b/270m are plain Gemma3ForCausalLM. We force ALL sizes to load as
  Gemma3ForCausalLM for text-only SFT. This is deliberate, not incidental:
  Gemma3ForConditionalGeneration.forward hand-rolls its loss with a per-microbatch
  nn.CrossEntropyLoss and sets accepts_loss_kwargs=False (transformers 4.53.2),
  so under gradient accumulation it mis-normalizes -- and it also produces a much
  worse base-model loss than the text submodel (4b wrapper ~14 vs 1b CausalLM ~2
  on step 1 of the same data; see the memory note / check_load_path.py). Loading
  as Gemma3ForCausalLM uses the correct self.loss_function path, drops the ~420M
  vision tower entirely (missing_keys=0; only vision_tower/multi_modal_projector
  land in unexpected_keys, which is what we want), and unifies every size onto the
  jlens-verified CausalLM path. Vision stays deferrable: the base tower is
  reattachable to any text checkpoint for the later multimodal recipe.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import transformers
from datasets import load_from_disk
from peft import LoraConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma3ForCausalLM,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from .chat import RESPONSE_TEMPLATE, ensure_chat_template
from .ckpt_schedule import LogSpacedCheckpointCallback

log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-name", default=None)

    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lr-scheduler", default="cosine")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--lora", action="store_true", help="LoRA instead of full fine-tuning (12b/27b).")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # Checkpointing for the interp-trajectory study. "log" saves at steps
    # {1,2,4,8,...}+final (dense early, where representations move fastest);
    # "uniform" is the stock every-N-steps behavior. See gemma_med.ckpt_schedule.
    p.add_argument("--ckpt-schedule", choices=["log", "uniform"], default="log")
    p.add_argument("--ckpt-log-base", type=float, default=2.0)
    p.add_argument("--save-steps", type=int, default=500, help="Only used with --ckpt-schedule uniform.")
    # The trajectory is the artifact for the interp analysis, so keep every
    # checkpoint by default; --save-only-model makes that affordable on disk.
    p.add_argument("--save-total-limit", type=int, default=None)
    p.add_argument("--save-only-model", action="store_true",
                   help="Drop optimizer/scheduler state: ~7x smaller checkpoints, but no resume.")

    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    # transformers emits: "It is strongly recommended to train Gemma3 models with
    # the `eager` attention implementation instead of `flash_attention_2`."
    # Gemma 3 mixes sliding-window and full-attention layers, and FA2 doesn't
    # honor every mask the model builds. We default to correctness; pass
    # --attn flash_attention_2 to trade it for speed once you've checked the
    # loss curves agree.
    p.add_argument("--attn", default="eager", choices=["flash_attention_2", "eager", "sdpa"])
    p.add_argument("--wandb-project", default="gemma-med-ft")
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--max-train-samples", type=int, default=None, help="Smoke-test escape hatch.")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    # Must be the {% generation %} variant: TRL builds the completion-only mask
    # from apply_chat_template(return_assistant_tokens_mask=True), which silently
    # finds zero assistant tokens without those markers. We swap back to the
    # stock template before saving so inference/vLLM get the official one.
    ensure_chat_template(tok, for_training=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Gemma pads on the right for training; left-padding is a generation concern.
    tok.padding_side = "right"

    cfg = AutoConfig.from_pretrained(args.model_path)
    is_wrapper = "Gemma3ForConditionalGeneration" in (cfg.architectures or [])
    log.info("model_type=%s architectures=%s -> load as Gemma3ForCausalLM (wrapper=%s)",
             cfg.model_type, cfg.architectures, is_wrapper)

    # Gemma 3 is bf16-native and A100s support it; fp16 overflows this family.
    # Force Gemma3ForCausalLM for every size (see module docstring): for 4b+ this
    # loads the text submodel from the multimodal checkpoint and drops the vision
    # tower, avoiding the wrapper's broken text-only loss path. Only the vision
    # weights should be missing from the *checkpoint's* perspective, so we assert
    # on missing_keys (LM weights that failed to load) and expect vision keys to
    # appear as unexpected -- exactly the head we want gone.
    Loader = Gemma3ForCausalLM if is_wrapper else AutoModelForCausalLM
    model, loading_info = Loader.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn,
        output_loading_info=True,
    )
    if type(model).__name__ != "Gemma3ForCausalLM":
        raise RuntimeError(f"expected Gemma3ForCausalLM, got {type(model).__name__}")
    missing = loading_info.get("missing_keys", [])
    if missing:
        raise RuntimeError(
            f"{len(missing)} weights missing after load (would be re-initialized): "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    unexpected = loading_info.get("unexpected_keys", [])
    if unexpected:
        dropped = sorted({k.split(".")[0] for k in unexpected})
        log.info("dropped %d non-LM weights (text-only): %s", len(unexpected), dropped)
    model.config.use_cache = False

    train = load_from_disk(str(Path(args.data_dir) / "train"))
    val = load_from_disk(str(Path(args.data_dir) / "val"))
    if args.max_train_samples:
        train = train.select(range(min(args.max_train_samples, len(train))))
    log.info("train=%d val=%d", len(train), len(val))

    peft_config = None
    if args.lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    sft_config = SFTConfig(
        output_dir=str(out),
        run_name=args.run_name or out.name,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_length=args.max_seq_len,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=200,
        # With "log", the callback owns saving via control.should_save, so the
        # default flow must not also save uniformly.
        save_strategy="no" if args.ckpt_schedule == "log" else "steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        report_to="wandb",
        seed=args.seed,
        # Train on assistant turns only -- the report's "completion only" loss.
        assistant_only_loss=True,
        dataset_num_proc=8,
    )

    callbacks = []
    if args.ckpt_schedule == "log":
        callbacks.append(LogSpacedCheckpointCallback(base=args.ckpt_log_base))

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tok,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    if trainer.is_world_process_zero():
        import wandb

        if wandb.run is not None:
            (out / "wandb_run_url.txt").write_text(wandb.run.url)
        (out / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    trainer.train()
    trainer.save_model(str(out / "final"))
    # Ship the stock template, not the training variant: {% generation %} is a
    # training-time artifact and shouldn't leak into the released checkpoint.
    ensure_chat_template(tok, for_training=False)
    tok.save_pretrained(str(out / "final"))
    log.info("saved -> %s", out / "final")


if __name__ == "__main__":
    main()
