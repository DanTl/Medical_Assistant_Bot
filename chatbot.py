# Interface to chat with a model.
#  Run:
#   python chat.py --model_path ./augmented_model
#

import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def start_chat(model_path: str):
    """
    Starts chat with model.
    """
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    except OSError:
        print("Error: Model not found.")
        return

    print("Hello, how can I help you?")

    while True:
        user_question = input("You: ")
        if user_question.lower() in ["quit", "exit"]:
            print("Assistant: have a great day!")
            break
        
        #Generate response
        prefixed_question = "answer the question: " + user_question
        inputs = tokenizer(prefixed_question, return_tensors="pt", max_length=128, truncation=True)
        
        out_ids = model.generate(
            inputs['input_ids'],
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
        
        assistant_response = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        print(f"Assistant: {assistant_response}")


if __name__ == "__main__":
    arg_processing = argparse.ArgumentParser(description="Chat with the medical assistant bot.")
    arg_processing.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to directory with the model."
    )
    args = arg_processing.parse_args()
    
    start_chat(model_path=args.model_path)