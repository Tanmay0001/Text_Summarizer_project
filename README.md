# Text_Summarizer_project
This project involves fine-tuning a Pegasus model on the SAMSum dataset for text summarization.

# Pegasus Summarization on SAMSum Dataset

## Project Overview

This project implements a sequence-to-sequence model using the Pegasus architecture for summarizing dialogues from the SAMSum dataset. The model is trained to generate concise summaries of conversational text, showcasing the capabilities of state-of-the-art natural language processing (NLP) techniques.

The SAMSum dataset consists of dialogues paired with their corresponding summaries, making it an ideal dataset for training and evaluating text summarization models.

## Dependencies
This project requires the following Python libraries:

torch: For PyTorch-based machine learning operations.
torchvision: For image processing (if needed).
torchaudio: For audio processing (if needed).
datasets: To load and preprocess datasets.
transformers: For pre-trained models and tokenization.
nltk: For natural language processing tasks, including tokenization.
tqdm: For displaying progress bars.
evaluate: For evaluation metrics.
rouge_score: For computing ROUGE scores.
py7zr: For handling 7z compressed files (if needed).

## Model Training
The Pegasus model is initialized using the pretrained checkpoint google/pegasus-cnn_dailymail.
The model is trained on the SAMSum dataset with evaluation using the ROUGE metric.
After training, the model and tokenizer are saved for later use.

## Evaluation
The model's performance is evaluated using ROUGE scores, indicating how closely the generated summaries match the reference summaries in the test dataset.
Example:
To see this model in action, you can run predictions on test samples after training. The script will display the input dialogue, the reference summary, and the generated summary by the model.


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Tanmay0001/Text_Summarizer_project.git
   cd Text_Summarizer_project
2. pip install -r requirements.txt

