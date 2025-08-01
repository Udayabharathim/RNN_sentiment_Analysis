# MOVIE REVIEW SENTIMENT ANALYSIS APPLICATION

## Problem Statement
In the digital age, movie reviews are flooded with opinions, ratings, and emotional responses from users across the globe. With such a massive influx of unstructured textual data, it becomes challenging for production companies, recommendation systems, and even casual users to automatically assess whether public sentiment is positive or negative.
Manually analyzing each review is time-consuming, subjective, and inefficient at scale.

Hence, there arises a need for an automated sentiment analysis system that can classify textual movie reviews into sentiment categories with high accuracy and real-time usability. This project aims to address that gap using Natural Language Processing and Deep Learning techniques.

## Overview 
This project is a Sentiment Analysis Web Application that uses a deep learning model to analyze the sentiment of movie reviews. It accepts a user-written review as input and predicts whether the sentiment behind the review is positive or negative.

The model is trained on the IMDB Movie Reviews Dataset, consisting of 50,000 labeled reviews. Text data is preprocessed using tokenization, padding, and embedding layers, followed by an LSTM-based neural network trained to understand the sequential nature of text. The trained model is then integrated into an interactive Gradio web interface, making it simple for any user to test reviews on-the-fly.

## Developed using:
- Python (Core language)
- TensorFlow / Keras for deep learning model
- Gradio for a simple and interactive web user interface


## FEATURES

- Accepts free-text movie reviews as input  
- Predicts the sentiment using a trained LSTM-based deep learning model  
- Friendly web UI using Gradio  
- Trained on the 50,000-review IMDB dataset for strong performance  
- Deployable and shareable as a public web app  

## PROJECT STRUCTURE

This repository contains the following important files:

1. Movie_Review.ipynb  
   → Jupyter Notebook containing code for preprocessing, tokenization, model building, and training.

2. Model_Testing_and_web_application.ipynb  
   → Code to test the trained model and build a Gradio-based web interface.

3. my_model.keras  
   → The saved trained deep learning model.

4. tokenizer.pkl  
   → Pickled tokenizer object used to vectorize input text for predictions.

5. IMDB Dataset.csv  
   → The dataset used for training, consisting of 50,000 movie reviews and their sentiment labels.

## HOW TO RUN LOCALLY

STEP 1: Clone the repository
-------------------------------------
Open your terminal or command prompt and run:

git clone https://github.com/your-username/movie-review-sentiment-app.git
cd movie-review-sentiment-app

STEP 2: Install dependencies
-------------------------------------
If you have a `requirements.txt` file, run:

pip install -r requirements.txt

OR manually install required libraries:

pip install gradio tensorflow pandas scikit-learn numpy

STEP 3: Run the application
-------------------------------------
Option 1: Launch via Python script (e.g., app.py if created)

python app.py

Option 2: Open and run the notebook
Open `Model_Testing_and_web_application.ipynb` in Jupyter Notebook or Google Colab and run all cells to launch the Gradio interface.

## MODEL INFORMATION

- Architecture: LSTM (Long Short-Term Memory Neural Network)
- Framework: TensorFlow / Keras
- Tokenizer: Pre-fitted tokenizer saved as `tokenizer.pkl`
- Output: Sentiment classification (Positive or Negative)
- Training Data: IMDB movie reviews dataset with 50k entries
- Input preprocessing: Lowercasing, padding, tokenizing
- Output display: Emoji-enhanced label via Gradio UI

## UI PREVIEW

<img width="1578" height="756" alt="image" src="https://github.com/user-attachments/assets/0c1d9a5f-2740-4d69-8665-cde3f41060c8" />
