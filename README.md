# Medical Assistant Bot

This repository contains a Medical Assistant Chatbot. The chatbot can answer medical questions based on a provided dataset. It's leveraging a fine-tuned T5 transformer model.

Two models were trained and compared:

1.  **Baseline Model:** Fine-tuned on the original cleaned dataset.
2.  **Augmented Model:** Fine-tuned on a larger dataset: original data + MedQuAD.

**Final Model:** The augmented model that included MedQuAD showed better performance.

## Project Structure
models/ # saved model files
  baseline_model
  augmented_model
data
  mle_screening_dataset.csv # Original dataset
  cleaned_medical_data.csv  # Processed clean data file
process_data.py # data cleaning, augmentation, and splitting
train.py # model training
run_evaluation.py # evaluating using ROUGE scores
chatbot.py # chat
requirements.txt
README.md


## Approach to the Problem

The requirement for this project was to build a question-answering system. Due to the fact that the systrem had to provide medical advices accuracy and reliability were number  one priorities. Thus, we chose a generative approach using a fine-tuned language model over a simpler retrieval-based system. It The trained model is able to understand the semantics of a user's question and gprovide a natural-language answer, not just matching keywords.

We chose  T5-small, a transfer transformer. T5 is apropriate because it frames every NLP problem as a text-to-text task, which is exatly what’s needed for question answering.

Here's the proposed workflow:

1.  **Data processing:**
    A critical part of this project was a data preprocessing. The raw dataset ontained significant formatting inconsistencies that made it unsuitable for direct model training. Many `answer` fields included non-answer text and instructional artifacts, such as:
    *   Parenthetical instructions - `Watch the video to learn more...`
    *   External links and references - `See this graphic for a quick overview...`
    *   Duplicated phrases within a single entry.
    
    Training the model on this noisy data would result in reproducing these junk phrases. The quality of responses would degrade. To solve this, a data cleaning was implemented in `process_data.py`. Regular expressions were used as the primary tool. For instance, the pattern `r'\s*\([^)]*\)'` was used to robustly find and eliminate any text enclosed in parentheses. The result was a clean and normalized dataset ready for the model training.

2.  **Data Augmentation:** To improve the model's generalization the dataset was augmented with the MedQuAD dataset from the Hugging Face Hub.

3.  **Model Training:** Two models were trained with the Hugging Face `Trainer` API:
    *   A **baseline model** was trained on the cleaned original dataset to have a performance benchmark.
    *   An **augmented model** was trained on combined original cleaned dataset and MedQuAD.
    Both of them were fine-tuned with 4 epochs and learning rate of 2e-5.

4.  **Model Evaluation:** The performance of the models was measured on test set originated from the original dataset. We chose the ROUGE metric for evaluation because it is the standard for measuring the quality of generated text. It's comparing word overlap ROUGE-1, ROUGE-2, and the longest common subsequence - ROUGE-L with a reference answer. ROUGE-L is very apropriate because it rewards structural similarity and coherence.

## Model Performance

From the results it's obvious that the augmented model outperformed the baseline model across all ROUGE metrics.

**Model A (Baseline)** 
ROUGE-1 Score: 23.04%         
ROUGE-2 Score: 11.78%          
ROUGE-L Score: 19.89%
                                            
**Model B (Augmented)**   


### Pros & Cons

**Pros:**
*   **Contextual Awarness:** The fine-tuned T5 model can understand paraphrased questions and provide relevant answers even if the wording doesn't exactly match the training data.
*   **Fluent Answers:** Produces natural, human-readable sentences, not just returning raw text from the dataset.
*   **Factually Grounded:** Training on reliable medical data including MedQuAD the answers are grounded in a trusted knowledge base.

**Cons:**
*   **Knowledge is limited:** The model's knowledge is limited by the training data. 
*   **Hallucination:** Small risk of generating factually incorrect information. It applies to all generative models. This might be a critical concern for  medical application.
*   **No Clarifying Questions:** The current architecture is a simple request-response model. Cannot esteblish dialogue or ask for more information.

## Potential Improvements and Extensions

*   **Refine Model Architecture:**
    *   **Using Larger Model:** Fine-tuning a larger base model like `t5-base` or FLAN-T5 should lead to more nuanced and accurate answers. The current model was chosen because of limited computational resources.
    *   **Domain-Specific Models:** Using a model pre-trained on spesific datalike BioBERT or BioGPT could potentially result inimproved understanding of medical terminology.

*   **Implement Guardrails and Safety Mechanisms:**
    *   **Retrieval Augmentation:** Combine the generative model with a retrieval system (like TF-IDF or vector search). The model could be required to base its answer on a retrieved document, reducing the risk of hallucination.
    *   **Confidence Scoring:** Estimate the model's confidence. If the confidence is below a certain threshold, we can program the bot to return "I can't provide an answer. Please consult a doctor."

*   **Improve User dilogue:**
    *   Introduce extended dilaugue logic to handle multi-turn conversations, ask clarifying questions, and maintain context.

*   **Augment the Dataset:**
    *   Keep augmenting the training data with more high-quality Q&A datasets.


## Instructions

1.  **Clone the repository:**
    git clone https://github.com/DanTl/medical-assistant-bot.git
  

2.  **Install dependencies:**
    * pip install -r requirements.txt
   

3.  **Prepare the data:** The first time you run a training script, the data will be automatically cleaned and saved.
    # (This step is handled automatically by the training script)


4.  **Train the models:**
    *   **Baseline:**
        python train.py --model_type baseline --output_dir ./models/baseline_model
    *   **Augmented with MedQuAD dataset:**
        python train.py --model_type augmented --output_dir ./models/augmented_model

5.  **Evaluate a model:**
    python evaluate.py --model_path ./models/augmented_model

6.  **Chat with the model:**
    python chat.py --model_path ./models/augmented_model

##Interactions with the trained model without augmentation

user: What are the symptoms of Glaucoma?
assistant: The Human Phenotype Ontology provides the following list of signs and symptoms for Glaucoma. If the information is available, the table below includes how often the symptom is seen in people with this condition. You can use the MedlinePlus Medical Dictionary to look up the definitions for these medical terms. Signs and Symptoms Approximate number of patients (when available) Abnormality of the symptom is unknown. Signs and Symptoms Approximate number of patients (when available) The Human Phenotype Ontology (HPO) has collected information on how often a sign or symptom occurs in a condition. The frequency may also be listed as a rough estimate of the percentage of patients who have that feature. The frequency may also be listed as a fraction. The first number of the fraction is how many people had the symptom, and the second number is the total number of people who were examined in one study. For example, a frequency of 25/25 means that in a study of 25 people all patients were found to have that symptom. The frequency may also be listed as a fraction. The first number of the fraction is how many people had the symptom, and the second number is the total number of people who were examined in one study. The frequency may also be listed as a fraction. The first number of the fraction is how many people had the symptom, and the second number is how many people had the symptom, and the second number is the total number of people who were examined in one study. The frequency may also be listed as a fraction. The first number of the fraction is how many people had the symptom, and the second number is how many people had the symptom, and the second number is the total number of people who were examined in one study. The frequency may also be listed as a fraction. The first number of the fraction is how many people had the symptom, and the second number is how many people had the symptom, and the second number is the total number of people who were examined in one study. The second number is how many people had the symptom, and the second number is the total number of people who were examined in one study. The second number is how many people had the symptom, and the second number is the total number of people who were examined

user: You: how is glaucoma treated?
assistant: The treatment of glaucoma is based on the treatment of glaucoma. The treatment of glaucoma is based on the treatment of glaucoma. The treatment of glaucoma is based on the treatment of glaucoma. The treatment of glaucoma is based on the treatment of glaucoma.

You: What is high blood pressure?
Bot: High blood pressure is a condition known as high blood pressure. It is a condition known as high blood pressure. It is a condition known as high blood pressure. It is a condition known as high blood pressure. It is a condition known as high blood pressure. It is a condition known as high blood pressure. It is a condition known as high blood pressure. It is a condition known as high blood pressure. It is a condition known as high blood pressure. It is a condition known as high blood pressure. The blood pressure is a high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure. High blood pressure is a condition known as high blood pressure

#Summary: 
Tendency for Repetitive Generation - During qualitative testing, the baseline model exhibited a tendency to get stuck in repetitive loops, a common failure mode for smaller generative models. It occurs because the model's limited vocabulary and shallow contextual understanding. This can be partially solved by adjusting text generation parameters but it's a fundamental weakness in the baseline model's ability to produce diverse long-form text.




