# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:32:38 2022

This file will load and train by using dataset of article and its 5 categories
- Sport
- Tech
- Business
- Entertainment
- Politics

@author: snaff
"""

import os
import datetime
import numpy as np
import pandas as pd
from article_categorization_module import ExploratoryDataAnalysis, ModelCreation 
from article_categorization_module import ModelEvaluation
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard


URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
TOKEN_SAVE_PATH = os.path.join(os.getcwd(),'Saved_path', 'tokenizer_data.json')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'Saved_path', 'model.h5')
LOG_PATH = os.path.join(os.getcwd(), 'Log')

# Constant
num_words = 50000
 
#%% EDA
# Step 1) load data
df = pd.read_csv(URL)

df.info()
# There is 2 columns named category and text
# Both with 2225 entries data and in object datatype
df.describe()
# duplicated data is spotted. Data cleaning is needed

df.duplicated().sum()
# There is 99 duplicated data, we should remove it and keep the first

# Step 2) Data Cleaning
eda = ExploratoryDataAnalysis()
df_clean = eda.clean_duplicate(df,keep='first')

article = df_clean['text']
category = df_clean['category']

# to calculate no of category
nb_categories = len(np.unique(category))

# remove tag and convert into lower case then split it in to element in list
article = eda.remove_tags(article)
article = eda.lower_split(article)

# Step 3) Features Selection

# Step 4) Data vectorization
# Tokenizer steps on the article_clean
article = eda.sentiment_tokenizer(article, TOKEN_SAVE_PATH,num_words=num_words)

# Pad sequence on the article   
article = eda.sentiment_pad_sequence(article)
        
# Step 5) Preprocessing
#One hot encoder
one_hot_encoder = OneHotEncoder(sparse=False)
category = one_hot_encoder.fit_transform(np.expand_dims(category,axis=-1))

# Train Test Split
# x = final_article, y = category
x_train, x_test, y_train, y_test = train_test_split(article, category,
                                                    test_size = 0.3,
                                                    random_state = 100)

# expand dimension into tensor
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# before fitting into model, need to convert x_train, x_test into float
x_train = np.asarray(x_train).astype('float32')
x_test = np.asarray(x_test).astype('float32')

# test the data to get the category matrix
print(y_train[0])
print(one_hot_encoder.inverse_transform(np.expand_dims(y_train[0], axis=0)))

# [1. 0. 0. 0. 0.] - business
# [0. 1. 0. 0. 0.] - entertainment
# [0. 0. 1. 0. 0.] - politics
# [0. 0. 0. 1. 0.] - sport
# [0. 0. 0. 0. 1.] - tech

#%% model creation
mc = ModelCreation()
model = mc.lstm_layer(num_words, nb_categories)

log_dir = os.path.join(LOG_PATH, 
                       datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

#%% Compile & model fitting
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='acc')

model.fit(x_train,y_train,epochs=10,
          validation_data=(x_test,y_test), 
          callbacks=tensorboard_callback)

#%% Model Evaluation
# Pre allocation of memory approach
predicted = np.empty([len(x_test), 5])
for index, test in enumerate(x_test):
    predicted[index,:] = model.predict(np.expand_dims(test,axis=0))

#%% Model analysis
y_pred = np.argmax(predicted, axis=1)
y_true = np.argmax(y_test, axis=1)

me = ModelEvaluation()
me.report_metrics(y_true, y_pred)

#%% Model Deployment
model.save(MODEL_SAVE_PATH)












