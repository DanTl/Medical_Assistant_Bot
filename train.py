import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from process_data import prepare_datasets

# Model
MODEL_CHECKPOINT = "t5-small"

def run_training(model_type: str, output_dir: str):
    """
    Model training and saving.
    """
    print(f"--- Starting Training ---")

    # process the datasets.
    use_augmentation = True if model_type == 'augmented' else False
    train_dataset, val_dataset, _ = prepare_datasets(use_augmentation=use_augmentation)

    # Set up the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    def tokenize_function(examples):
        questions = [str(q) for q in examples['question']]
        answers = [str(a) for a in examples['answer']]
        inputs = ["answer the question: " + q for q in questions]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(text_target=answers, max_length=512, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Load the pre-trained model and set up training arguments.
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=4,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start the training process.
    trainer.train()
    print("Training finished.")

    # Saving fine-tuned model.
    trainer.save_model(output_dir)
    print("Model saved successfully.")

if __name__ == "__main__":
    pars = argparse.ArgumentParser(description="Train a T5 model for the medical assistant.")
    pars.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=['baseline', 'augmented'],
        help="Specify which model to train: 'baseline' or 'augmented'."
    )
    pars.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory where the trained model will be saved."
    )
    args = pars.parse_args()
    
    run_training(model_type=args.model_type, output_dir=args.output_dir)