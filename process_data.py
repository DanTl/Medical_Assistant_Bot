# Contains all the logic for loading, cleaning, augmenting,
# and preparing the datasets for the medical chatbot project.

import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import os

ORIGINAL_DATA_PATH = "data/mle_screening_dataset.csv"
CLEANED_DATA_PATH = "data/cleaned_medical_data.csv" # Path to save/load the clean file
RANDOM_SEED = 42

def restructure_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset using string truncation method.
    """
    new_lines = []
    junk_pieces = [
        "(Watch the video", "See this graphic", "See a glossary",
        "Get tips on finding", "Learn what a comprehensive"
    ]

    for index, row in df.iterrows():
        original_question = str(row['question'])
        answer_text = str(row['answer'])
        
        min = -1
        for junk in junk_pieces:
            junk_index = answer_text.find(junk)
            if junk_index != -1:
                if min == -1 or junk_index < min:
                    min = junk_index
        
        if min != -1:
            clean_text = answer_text[:min]
        else:
            clean_text = answer_text
        
        final_clean_answer = " ".join(clean_text.split()).strip()
        
        if original_question and final_clean_answer:
            new_lines.append({
                'question': original_question,
                'answer': final_clean_answer
            })
            
    updated_df = pd.DataFrame(new_lines)

    return updated_df


def prepare_datasets(use_augmentation: bool = False):
    """
    Prepares the data for training.
    Checks for a pre-cleaned file before running the full cleaning process.
    """

    # Check if a clean version of the data already exists
    if os.path.exists(CLEANED_DATA_PATH):
        print(f"Found precleaned data file: {CLEANED_DATA_PATH}. Loading...")
        df_cleaned = pd.read_csv(CLEANED_DATA_PATH)
    else:
        print("No pre-cleaned data file found. Running the full process...")
        # Load the raw data from the CSV file.
        df_original_raw = pd.read_csv(ORIGINAL_DATA_PATH)
        # Call the cleaning function.
        df_cleaned = restructure_data(df_original_raw)
        # Manage duplicates and missing values.
        df_cleaned.drop_duplicates(subset=['question'], inplace=True)
        df_cleaned.dropna(inplace=True)
        # Save the cleaned data
        df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)

    result_df = df_cleaned

    if use_augmentation: # Add MedQuAD dataset
        dataset_medquad = load_dataset("lavita/MedQuAD", split="train")
        dataset_medquad = dataset_medquad.rename_column("Question", "question")
        dataset_medquad = dataset_medquad.rename_column("Answer", "answer")
        dataset_medquad = dataset_medquad.remove_columns([col for col in dataset_medquad.column_names if col not in ["question", "answer"]])
        df_medquad = dataset_medquad.to_pandas()
        combined_df = pd.concat([df_cleaned, df_medquad], ignore_index=True)
        combined_df.drop_duplicates(subset=['question'], inplace=True)
        result_df = combined_df

    # Splitting
    _, test_df = train_test_split(df_cleaned, test_size=0.1, random_state=RANDOM_SEED)
    train_val_df = result_df[~result_df['question'].isin(test_df['question'])]
    train_df, val_df = train_test_split(train_val_df, test_size=0.1, random_state=RANDOM_SEED)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    return train_dataset, val_dataset, test_dataset