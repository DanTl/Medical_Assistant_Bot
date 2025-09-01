import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from process_data import prepare_datasets
import evaluate

def run_evaluation(model_path: str):
    """
    evaluates model on the test set using ROUGE scores.
    """
    print("Starting Evaluation")
    
    #Check for GPU availability ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    #Load the fine-tuned model and tokenizer.
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    #Load test data.
    _, _, test_dataset = prepare_datasets(use_augmentation=False)

    #Generating predictions.
    model_predictions = []
    references = []
    for item in tqdm(test_dataset):
        question = "answer the question: " + str(item['question'])
        inputs = tokenizer(question, return_tensors="pt", max_length=128, truncation=True)
        
        #Move the input data to the same device as the model
        in_ids = inputs.input_ids.to(device)
        
        out_ids = model.generate(in_ids, max_length=512, num_beams=5, early_stopping=True)
        predict_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        
        model_predictions.append(predict_text)
        references.append(str(item['answer']))

    #Compute ROUGE scores.
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=model_predictions, references=references)

    print(f"  ROUGE-1: {results['rouge1'] * 100:.2f}%")
    print(f"  ROUGE-2: {results['rouge2'] * 100:.2f}%")
    print(f"  ROUGE-L: {results['rougeL'] * 100:.2f}%")


if __name__ == "__main__":
    arg_processing = argparse.ArgumentParser(description="Evaluate a trained T5 model.")
    arg_processing.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the directory containing the saved model."
    )
    args = arg_processing.parse_args()
    
    run_evaluation(model_path=args.model_path)