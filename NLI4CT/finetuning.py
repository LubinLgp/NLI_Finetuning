import json
import sys
import argparse
from pathlib import Path
from unittest.mock import MagicMock
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# --- Mock bitsandbytes (Inchangé) ---
BNB_AVAILABLE = False
def create_bnb_mock():
    class MockLinear8bitLt: pass
    class MockBNBModule:
        class nn: Linear8bitLt = MockLinear8bitLt; Linear4bit = MockLinear8bitLt
        class optim: GlobalOptimManager = MagicMock
    mock_bnb = MockBNBModule()
    sys.modules['bitsandbytes'] = mock_bnb
    sys.modules['bitsandbytes.nn'] = mock_bnb.nn
    sys.modules['bitsandbytes.optim'] = mock_bnb.optim
    sys.modules['bitsandbytes.cextension'] = MagicMock()
    return mock_bnb

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except:
    BNB_AVAILABLE = False
    create_bnb_mock()

# --- Fonctions utilitaires ---
def load_jsonl(path: Path):
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): samples.append(json.loads(line))
    return samples

def build_dataset_from_messages(path: Path, tokenizer):
    raw_samples = load_jsonl(path)
    texts = [{"text": tokenizer.apply_chat_template(s["messages"], tokenize=False)} for s in raw_samples]
    return Dataset.from_list(texts)


def _create_model(model_name: str, bnb_config: BitsAndBytesConfig):
    """Create quantized model and prepare for k-bit training (for model_init / final run)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return prepare_model_for_kbit_training(model)


class OptunaProgressCallback(TrainerCallback):
    """Affiche le numéro d'essai au début et à la fin de chaque trial Optuna."""
    def __init__(self, n_trials):
        self.n_trials = n_trials
        self.trial_num = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.trial_num += 1
        print(f"\n{'='*60}\n  Optuna trial {self.trial_num}/{self.n_trials} started\n{'='*60}\n", flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        print(f"\n  >>> Optuna trial {self.trial_num}/{self.n_trials} finished <<<\n", flush=True)


def _optuna_hp_space(trial, search_epochs_max=4):
    """Optuna search space. Use short epochs (2--search_epochs_max) during search to save time; final run uses --final_epochs."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [2, 4, 8]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, search_epochs_max),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [2, 4, 8]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_file", type=str, default="NLI4PR_train.jsonl")
    parser.add_argument("--eval_file", type=str, default=None, help="Eval JSONL for Optuna. If not set and use_optuna, 10%% of train is used as eval.")
    parser.add_argument("--use_optuna", action="store_true", help="Run Optuna hyperparameter search before final training.")
    parser.add_argument("--n_trials", type=int, default=6, help="Number of Optuna trials (default 6). For ~7h/run and 72h limit, use 6--8 trials with short search epochs.")
    parser.add_argument("--optuna_epochs_max", type=int, default=4, help="Max epochs per Optuna trial (default 4). Short trials = faster search; final run uses --final_epochs.")
    parser.add_argument("--final_epochs", type=int, default=5, help="Epochs for the final training after Optuna (default 5). Used with --use_optuna or --load_optuna_hp.")
    parser.add_argument("--optuna_save_best_hp", type=str, default=None, help="With --use_optuna: run only the search and save best HP to this JSON file (no final training). For Job 1 in two-job strategy.")
    parser.add_argument("--load_optuna_hp", type=str, default=None, help="Load best HP from this JSON and run only final training (no search). For Job 2 in two-job strategy. Use --eval_file for eval during training.")
    args = parser.parse_args()

    MODEL_NAME = args.model_path
    OUTPUT_DIR = Path(args.output_dir)
    TRAIN_PATH = Path(args.train_file)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    train_dataset = build_dataset_from_messages(TRAIN_PATH, tokenizer)

    if args.use_optuna or args.load_optuna_hp:
        if args.eval_file and Path(args.eval_file).exists():
            eval_dataset = build_dataset_from_messages(Path(args.eval_file), tokenizer)
        elif args.use_optuna:
            split = train_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split["train"]
            eval_dataset = split["test"]
            print(f"Optuna: using 10% of train as eval ({len(eval_dataset)} examples).")
        else:
            eval_dataset = None  # load_optuna_hp without eval_file: no eval during final run
    else:
        eval_dataset = None

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        disable_tqdm=False,
        log_level="info",
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="none",
        ddp_find_unused_parameters=False,
        eval_strategy="epoch" if eval_dataset else "no",
    )

    # --- Job 2: load saved HP and run only final training ---
    if args.load_optuna_hp:
        hp_path = Path(args.load_optuna_hp)
        if not hp_path.exists():
            raise FileNotFoundError(f"Hyperparameters file not found: {hp_path}")
        with hp_path.open("r", encoding="utf-8") as f:
            best_hp = json.load(f)
        for key, value in best_hp.items():
            if hasattr(training_args, key):
                setattr(training_args, key, value)
        training_args.num_train_epochs = args.final_epochs
        print(f"Loaded best hyperparameters from {hp_path}. Running final training ({args.final_epochs} epochs).")
        model = _create_model(MODEL_NAME, bnb_config)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            dataset_text_field="text",
            peft_config=lora_config,
        )
        trainer.train()
        trainer.model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"\nDone. Model saved to {OUTPUT_DIR}")
        return

    if args.use_optuna:
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for --use_optuna. Install with: pip install optuna")

        def model_init(trial):
            return _create_model(MODEL_NAME, bnb_config)

        # SFTTrainer (trl) accède à model dans __init__, donc on ne peut pas passer model=None.
        # On passe un modèle initial pour l'init ; hyperparameter_search appellera model_init(trial) à chaque essai.
        initial_model = model_init(None)

        trainer = SFTTrainer(
            model=initial_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            dataset_text_field="text",
            peft_config=lora_config,
            model_init=model_init,
            callbacks=[OptunaProgressCallback(args.n_trials)],
        )

        def hp_space(trial):
            return _optuna_hp_space(trial, search_epochs_max=args.optuna_epochs_max)

        print(f"\nOptuna: {args.n_trials} trials (max {args.optuna_epochs_max} epochs each), then final run with {args.final_epochs} epochs.")
        print(f"Train: {len(train_dataset)} / Eval: {len(eval_dataset)}")
        best_trials = trainer.hyperparameter_search(
            direction="minimize",
            backend="optuna",
            hp_space=hp_space,
            n_trials=args.n_trials,
            compute_objective=lambda metrics: metrics.get("eval_loss", float("inf")),
        )
        best_run = best_trials[0] if isinstance(best_trials, list) else best_trials
        for key, value in best_run.hyperparameters.items():
            setattr(training_args, key, value)
        # Final run always uses --final_epochs (full training), not the short trial epochs
        training_args.num_train_epochs = args.final_epochs
        print("Best hyperparameters (final run will use --final_epochs=%d):" % args.final_epochs, best_run.hyperparameters)

        # --- Job 1: only save best HP to file and exit (no final training) ---
        if args.optuna_save_best_hp:
            hp_path = Path(args.optuna_save_best_hp)
            hp_path.parent.mkdir(parents=True, exist_ok=True)
            # Convert to native Python types for JSON (Optuna may return numpy scalars)
            best_hp = {}
            for k, v in best_run.hyperparameters.items():
                if hasattr(v, "item"):
                    best_hp[k] = v.item()
                else:
                    best_hp[k] = v
            best_hp["num_train_epochs"] = args.final_epochs
            with hp_path.open("w", encoding="utf-8") as f:
                json.dump(best_hp, f, indent=2)
            print(f"\nOptuna done. Best hyperparameters saved to {hp_path} (no final training; run Job 2 with --load_optuna_hp {hp_path})")
            return

        # Final training with best hyperparameters
        print("\nTraining final model with best hyperparameters...")
        model = _create_model(MODEL_NAME, bnb_config)
        trainer_final = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            dataset_text_field="text",
            peft_config=lora_config,
        )
        trainer_final.train()
        trainer_final.model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"\nDone. Model saved to {OUTPUT_DIR}")
    else:
        model = _create_model(MODEL_NAME, bnb_config)
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            peft_config=lora_config,
            args=training_args,
        )
        print(f"\nStarting training on {len(train_dataset)} examples.")
        trainer.train()
        trainer.model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"\nDone. Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
