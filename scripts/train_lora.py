"""
LoRA fine-tune of Qwen2.5-0.5B-Instruct on `data/dataset/train.jsonl` → `models/navi-lora/`.

Runs on Apple MPS (an M2 with 8 GB, ~20–30 minutes for one epoch), CUDA (Colab T4) or CPU — the same script.

LoRA (Low-Rank Adaptation): the 494M base weights stay frozen; next to every linear layer two
small matrices A (r × d) and B (d × r) are trained and their product is added to the layer's
output. With r = 16 that is ~8.8M trainable parameters (1.8 %) — it fits in 8 GB and the
adapter file is ~35 MB instead of a 1 GB model copy.

    python scripts/train_lora.py [--epochs 1] [--out models/navi-lora]

On a laptop run it under `caffeinate -i` (macOS): a Mac that sleeps mid-run pauses the GPU, and
the first attempt here crawled from 20 s to 700 s per step before dying in an out-of-memory.
"""

import argparse
import json
import random
import time

import torch
from common import ROOT
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from navi.services.generator import pick_device

BASE = 'Qwen/Qwen2.5-0.5B-Instruct'


def load(path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding='utf-8')]


def main() -> None:
    parser = argparse.ArgumentParser()
    # One epoch: the train loss is flat (~0.3) from half an epoch on, and on 8 GB a second epoch
    # costs the time without buying anything measurable.
    parser.add_argument('--epochs', type=float, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--out', default=str(ROOT / 'models' / 'navi-lora'))
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    device = pick_device()
    rows = load(ROOT / 'data' / 'dataset' / 'train.jsonl')
    # 10 % held out from training: eval loss shows over-fitting while training runs.
    random.Random(1).shuffle(rows)
    n_val = max(16, len(rows) // 10)
    val, train = rows[:n_val], rows[n_val:]
    fields = ('prompt', 'completion')
    train_ds = Dataset.from_list([{k: r[k] for k in fields} for r in train])
    val_ds = Dataset.from_list([{k: r[k] for k in fields} for r in val])

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    lengths = [
        # transformers 5 returns a BatchEncoding (input_ids + attention_mask), not a list.
        len(
            tokenizer.apply_chat_template(r['prompt'] + r['completion'], tokenize=True)['input_ids']
        )
        for r in rows
    ]
    print(
        f'device={device} train={len(train)} val={len(val)} tokens max={max(lengths)} '
        f'mean={sum(lengths) // len(lengths)}'
    )

    # float32 for training: fp16 on MPS has no loss scaling and diverges; bf16 is slow there.
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        # Every linear layer of the blocks (attention q/k/v/o and the MLP): Jaydari trained only
        # q_proj and v_proj — cheaper, but a format/behaviour change sits in the MLP too.
        target_modules='all-linear',
        task_type='CAUSAL_LM',
    )
    config = SFTConfig(
        output_dir=str(ROOT / 'models' / 'checkpoints'),
        num_train_epochs=args.epochs,
        # Batch 1 × accumulation 8 = effective batch 8. The logits of one example are
        # 650 tokens × 151,936 vocabulary × 4 bytes ≈ 0.4 GB, and backward doubles it:
        # batch 2 pushed an 8 GB machine into swap (a step went from 20 s to 700 s).
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        # The default eval batch is 8 — eight sets of logits at once ran MPS out of memory.
        per_device_eval_batch_size=1,
        learning_rate=args.lr,
        lr_scheduler_type='cosine',
        # transformers 5: a float below 1 is a ratio of the steps (warmup_ratio was removed).
        warmup_steps=0.05,
        max_length=max(lengths) + 8,
        # The loss covers the assistant's answer only: the model is not trained to write sources.
        completion_only_loss=True,
        gradient_checkpointing=True,
        logging_steps=5,
        eval_strategy='no',  # one evaluate() after training; mid-run eval doubled the peak
        save_strategy='no',
        report_to='none',
        seed=42,
        bf16=False,
        fp16=False,
        dataloader_pin_memory=False,  # pinned memory is a CUDA feature; MPS warns and ignores it
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.model.print_trainable_parameters()
    t0 = time.perf_counter()
    result = trainer.train()
    minutes = (time.perf_counter() - t0) / 60
    # Save first: an out-of-memory in evaluate() must not throw away the trained adapter.
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    metrics = trainer.evaluate()
    log = {
        'base_model': BASE,
        'device': device,
        'train_examples': len(train),
        'val_examples': len(val),
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'lora_rank': args.rank,
        'train_loss': round(result.training_loss, 4),
        'eval_loss': round(metrics['eval_loss'], 4),
        'minutes': round(minutes, 1),
        'history': [h for h in trainer.state.log_history if 'loss' in h or 'eval_loss' in h],
    }
    (ROOT / 'reports').mkdir(exist_ok=True)
    (ROOT / 'reports' / 'training.json').write_text(json.dumps(log, indent=2))
    print(json.dumps({k: v for k, v in log.items() if k != 'history'}, indent=2))


if __name__ == '__main__':
    main()
