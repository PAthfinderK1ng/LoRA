import os
import time
import warnings

import numpy as np
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer,
	TrainingArguments,
	Trainer,
	default_data_collator,
)



def compute_metrics(eval_pred) -> dict:
	logits = eval_pred.predictions if hasattr(eval_pred, "predictions") else eval_pred[0]
	labels = eval_pred.label_ids if hasattr(eval_pred, "label_ids") else eval_pred[1]
	preds = np.argmax(logits, axis=-1)
	accuracy = (preds == labels).mean().item()
	return {"accuracy": accuracy}


def sanitize_module_tree(module: torch.nn.Module) -> None:
	modules = module._modules
	if not isinstance(modules, dict):
		try:
			modules = dict(modules)
		except Exception:
			modules = {}
	cleaned = {}
	for name, child in list(modules.items()):
		if not isinstance(name, str):
			continue
		if not isinstance(child, torch.nn.Module):
			continue
		cleaned[name] = child
	module._modules = cleaned
	for child in cleaned.values():
		sanitize_module_tree(child)


def main() -> None:
	warnings.filterwarnings("ignore", category=FutureWarning)
	warnings.filterwarnings("ignore", message="`resume_download` is deprecated")
	os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
	os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
	os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

	ds = load_dataset("glue", "sst2")
	label_names = ds["train"].features["label"].names
	sample = ds["train"][0]

	run_id = time.strftime("%Y%m%d_%H%M%S")
	output_dir = f"./outputs/{run_id}"
	adapter_dir = f"{output_dir}/lora_adapter"

	model_name = "bert-base-uncased"
	_ = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
	_ = AutoTokenizer.from_pretrained(model_name)
	max_length = 128
	padding = "max_length"
	full_finetune = True
	
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	def preprocess(example: dict) -> dict:
		encoding = tokenizer(
			example["sentence"],
			max_length=max_length,
			padding=padding,
			truncation=True,
		)
		encoding["labels"] = int(example["label"])
		return encoding

	tokenized_dataset = ds.map(preprocess, remove_columns=ds["train"].column_names)

	def to_torch(example: dict) -> dict:
		return {key: torch.tensor(value) for key, value in example.items()}

	tokenized_dataset.set_transform(to_torch)

	train_limit = min(10000000, len(tokenized_dataset["train"]))
	eval_limit = min(1000, len(tokenized_dataset["validation"]))
	train_dataset = tokenized_dataset["train"].select(range(train_limit))
	eval_dataset = tokenized_dataset["validation"].select(range(eval_limit))
	batch_size = 16
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	batch = next(iter(train_loader))

	model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

	if full_finetune:
		print("Full fine-tune baseline (no LoRA)")
	else:
		lora_config = LoraConfig(
			r=8,
			lora_alpha=16,
			lora_dropout=0.1,
			task_type=TaskType.SEQ_CLS,
			target_modules=["query", "value"],
			modules_to_save=["classifier"],
		)
		model = get_peft_model(model, lora_config)
		sanitize_module_tree(model)
		model.print_trainable_parameters()

	training_args = TrainingArguments(
		output_dir=output_dir,
		learning_rate=2e-4,
		per_device_train_batch_size=16,
		per_device_eval_batch_size=16,
		num_train_epochs=3,
		evaluation_strategy="epoch",
		save_strategy="no",
		logging_strategy="epoch",
		logging_first_step=False,
		load_best_model_at_end=False,
		metric_for_best_model="accuracy",
		label_names=["labels"],
		remove_unused_columns=False,
		max_grad_norm=0.0,
		fp16=torch.cuda.is_available(),
	)
	print(
		f"Training config: lr={training_args.learning_rate}, "
		f"epochs={training_args.num_train_epochs}, "
		f"batch={training_args.per_device_train_batch_size}"
	)

	trainer = Trainer(
		model=model,
		tokenizer=tokenizer,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		data_collator=default_data_collator,
		compute_metrics=compute_metrics,
	)
	print(">>> training start")
	try:
		trainer.train()
		print(">>> training done")
	except Exception as exc:
		print(">>> training failed:", repr(exc))
		raise
	final_metrics = trainer.evaluate()
	print("final metrics:", final_metrics)
	if not full_finetune:
		adapter_dir = f"{output_dir}/lora_adapter"
		trainer.model.save_pretrained(adapter_dir)
		tokenizer.save_pretrained(adapter_dir)
		print("saved adapter to:", adapter_dir)

		base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
		inference_model = PeftModel.from_pretrained(base_model, adapter_dir)
		inference_model.eval()
		texts = ["I love this movie.", "This is the worst film ever."]
		encoded = tokenizer(
			texts,
			max_length=max_length,
			padding=padding,
			truncation=True,
			return_tensors="pt",
		)
		encoded = {key: value.to(inference_model.device) for key, value in encoded.items()}
		with torch.no_grad():
			outputs = inference_model(**encoded)
			preds = torch.argmax(outputs.logits, dim=-1).tolist()
		label_map = {idx: name for idx, name in enumerate(label_names)}
		for text, pred in zip(texts, preds):
			print("text:", text, "->", label_map.get(pred, pred))
	do_forward = True
	if do_forward:
		model.eval()
		batch_on_device = {
			key: value.to(model.device) if torch.is_tensor(value) else value
			for key, value in batch.items()
		}
		with torch.no_grad():
			outputs = model(**batch_on_device)
		print("loss:", float(outputs.loss))
		print("logits shape:", tuple(outputs.logits.shape))


if __name__ == "__main__":
	main()