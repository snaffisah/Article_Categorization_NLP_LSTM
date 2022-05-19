# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:32:29 2022

This file is to test the new input article and predict the category

@author: snaff
"""

import os
import json
import numpy as np
from article_categorization_module import ExploratoryDataAnalysis
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

TOKEN_SAVE_PATH = os.path.join(os.getcwd(),'Saved_path', 'tokenizer_data.json')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'Saved_path', 'model.h5')

#%%
article_classifier = load_model(MODEL_SAVE_PATH)
article_classifier.summary()

#%% Tokenizer loading
with open(TOKEN_SAVE_PATH, 'r') as json_file:
    token = json.load(json_file)

#%% Deploy
# Step 1) Loading of data
new_article = [input('Please copy the article from news here:\n\n')]
            
# Step 2) Data cleaning
eda = ExploratoryDataAnalysis()
new_article = eda.remove_tags(new_article)
new_article = eda.lower_split(new_article)

# Step 3) Feature selection
# Step 4) Data preprocessing

loaded_tokenizer = tokenizer_from_json(token)
new_article = loaded_tokenizer.texts_to_sequences(new_article)
new_article = eda.sentiment_pad_sequence(new_article)

#%% model prediction
outcome = article_classifier.predict(new_article)
print('\n', outcome) # only give num result

category_label = ['business', 'entertainment', 'politics', 'sport', 'tech']
print('\nThe article category is ' + category_label[np.argmax(outcome)])

# [1. 0. 0. 0. 0.] - business
# [0. 1. 0. 0. 0.] - entertainment
# [0. 0. 1. 0. 0.] - politics
# [0. 0. 0. 1. 0.] - sport
# [0. 0. 0. 0. 1.] - tech
