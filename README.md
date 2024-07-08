# Ligthweight_finetuning_to_-foundation_model

Project Overview
This project demonstrates parameter-efficient fine-tuning of pre-trained language models using the Hugging Face transformers library and PyTorch. The goal is to adapt a foundation model for sentiment analysis of customer reviews while minimizing computational resources. The project compares the performance of three models:

* The original pre-trained model.
* A fine-tuned model.
* A PEFT (Parameter-Efficient Fine-Tuning) models (v1 & v2).

Project Structure
* data/model:  Directory where fine-tuned model checkpoints and logs are saved.
* data/peft_model_v1:  Directory where fine-tuned model checkpoints and logs are saved.
* data/peft_model_v1:  Directory where fine-tuned model checkpoints and logs are saved.
* logs: Directory containing the logs.
* images: Directory containing images used in the notebook
* iypnb file: Jupyter notebooks for experimentation and visualization.
* README.md: Project documentation.

Requirements
To run the project, you need the following libraries:

* Python 3.6+
* transformers
* torch
* datasets
* pandas
* numpy
* matplotlib
* re
* ace_tools (for displaying DataFrames in notebooks)

Install the required libraries using pip:

bash
Copy code
pip install transformers torch datasets pandas numpy matplotlib ace_tools

Dataset
The project uses a dataset that consists of tweets related to financial news, each labeled with a sentiment class (bullish, bearish, neutral).

Preprocessing and Tokenization
The dataset is preprocessed to:

* Convert text to lowercase.
* Remove special characters.
* Tokenization is performed using the distilbert-base-cased tokenizer from Hugging Face.

Models
1. Original Pre-trained Model: The pre-trained distilbert-base-cased model is loaded and evaluated on the test dataset without any fine-tuning.

2. Fine-Tuned Model: The pre-trained model is fine-tuned on the training dataset for sentiment analysis. Fine-tuning adjusts all the model parameters to better fit the specific task.

3. PEFT Models (v2 & v2): Parameter-efficient fine-tuning is performed on the pre-trained model, adjusting only a small subset of parameters while keeping the rest of the model fixed. This approach reduces computational resource requirements and prevents overfitting.

Evaluation
All three models are evaluated using the test dataset. 
The evaluation metrics include:

* Loss
* Accuracy
* Eval Runtime
* F1 Score
* Precission
* Recall

The results are compared to determine the effectiveness of fine-tuning and parameter-efficient fine-tuning.
