from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = Path("data/fine_tuning")
OUTPUT_DIR = "models/qwen2.5-1.5b-clauseguard"


def format_example(example: dict, tokenizer) -> dict:
    """Format one JSONL chat example using the model's chat template."""

    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(DATA_DIR / "train.jsonl"),
            "eval": str(DATA_DIR / "eval.jsonl"),
        },
    )

    dataset = dataset.map(
        lambda example: format_example(example, tokenizer)
    )

    use_gpu = torch.cuda.is_available()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if use_gpu else torch.float32,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none",
        max_length=1024,
        dataset_text_field="text",
        use_cpu=not use_gpu,
        fp16=use_gpu,
        bf16=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Fine-tuned model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()