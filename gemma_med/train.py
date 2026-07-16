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
* Gemma 3 4b+ are Gemma3ForConditionalGeneration (a multimodal wrapper) even for
  text-only use; 1b is Gemma3ForCausalLM. We load AutoModelForCausalLM and let
  transformers pick, then train the language model only.
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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from .chat import RESPONSE_TEMPLATE, ensure_chat_template

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

    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--attn", default="flash_attention_2", choices=["flash_attention_2", "eager", "sdpa"])
    p.add_argument("--wandb-project", default="gemma-med-ft")
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
    log.info("model_type=%s architectures=%s", cfg.model_type, cfg.architectures)

    # Gemma 3 is bf16-native and A100s support it; fp16 overflows this family.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn,
    )
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
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="wandb",
        seed=args.seed,
        # Train on assistant turns only -- the report's "completion only" loss.
        assistant_only_loss=True,
        dataset_num_proc=8,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train,
        eval_dataset=val,
        processing_class=tok,
        peft_config=peft_config,
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
