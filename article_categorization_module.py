# -*- coding: utf-8 -*-
"""
Created on Thu May 19 10:24:39 2022

This file contain the needed modules to run the train file

@author: snaff
"""

import re
import json
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Bidirectional, Embedding
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score


class ExploratoryDataAnalysis():
    
    def __init__(self):
        pass

    def clean_duplicate(self, data, keep='first'):
        '''
        To drop duplicated and leave the fist duplicated data

        Parameters
        ----------
        data : Array
            Raw data

        Returns
        -------
        data : Array
            data with no duplicate

        '''
        data = data.drop_duplicates(keep='first')
            
        return data

    def remove_tags(self,data):
        '''
        To remove the html function and return review in series

        Parameters
        ----------
        data : Array
            Raw training data containing strings

        Returns
        -------
        data : Array
            Clean all data, without html function

        '''
        
        data = [re.sub('<.*?>', '', text) for text in data]
            
        return data
    
    def lower_split(self,data):
        '''
        This function converts all letters into lowercase and split into list.
        Also filtered numerical data.

        Parameters
        ----------
        data : Array
            Cleaned training data containing strings

        Returns
        -------
        data : Array
            Cleaned all data that converted into lowercase

        '''
       
        data = [re.sub('[^a-zA-Z]', ' ', text).lower().split() for text in data]
        
        return data
    
    def sentiment_tokenizer(self,data,token_save_path,
                            num_words=10000, oov_token='<OOV>', prt=False):
        '''
        This function will collect each of the sentiment word according to the 
        limit set and save it desending from the most frequently appeared 
        words and ignore the rest

        Parameters
        ----------
        data : Array
            Cleaned training data
        token_save_path : cwd
            Save data in current working directory in JSON format
        num_words : int, optional
            The limit of token words need to consider. The default is 10000.
        oov_token : The default is '<OOV>'
            Out of vacabolary words. Will be ignored and set value as 1.
        prt : Boolean
            To print the token words. The default is False.

        Returns
        -------
        data : Dict
            Return the dictionary of the token in ascending order

        '''
        
        # tokenizer to vectorize the words
        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        tokenizer.fit_on_texts(data)

        # To save the tokenizer for deployment purpose
        token_json = tokenizer.to_json()

        with open(token_save_path,'w') as json_file:
            json.dump(token_json, json_file)

        # to observe the number of words
        word_index = tokenizer.word_index
        
        if prt == True:
            # to view the tokenized words
            # print(word_index)
            print(dict(list(word_index.items())[0:10]))

        # to vectorize the sequences of text
        data = tokenizer.texts_to_sequences(data)
        
        return data
        
    def sentiment_pad_sequence(self,data):
        '''
        This function padding the token words and the sentiment together and
        makesure all data in same length. If exceed, it will be ignored

        Parameters
        ----------
        data : Array
            Cleaned training data

        Returns
        -------
        data: Array
            Paddied of training data and its sentiment

        '''
        
        return pad_sequences(data, maxlen=500, padding='post',
                             truncating='post')
    
class ModelCreation():
    
    def __init__(self):
        pass
    
    def lstm_layer(self,num_words, nb_categories, embedding_output=128, 
                   nodes=64, dropout=0.2):
        '''
        This function is to creates a LSTM model with 2 hidden layers. 
        Last layer of the model comrises of softmax activation function
     
        Parameters
        ----------
        num_words:Int
        nb_categories: Int
            Contains the lenght of unique sentiment
        embedding output: Int
            DESCRIPTION. The default is 128
        nodes : Int, optional
            DESCRIPTION. The default is 64
        dropout : Float, optional
            DESCRIPTION. The default is 0.2
     
        Returns
        -------
        Model: Created Model

        '''
        model = Sequential()
        model.add(Embedding(num_words, embedding_output))
        #model.add(Bidirectional(LSTM(nodes,return_sequences=True)))
        #model.add(Dense(nodes, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dense(nodes, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories, activation='softmax'))
        model.summary()
        
        return model
    
    
class ModelEvaluation():
    def report_metrics(self, y_true, y_pred):
        '''
        This function is to evaluate the model created. 
        1. Classification report
        2. Confusion matrix
        3. Accuracy score

        Parameters
        ----------
        y_true : Array
            True value in array
        y_pred : Array
            Prediction value in array

        Returns
        -------
        None.

        '''
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        print(accuracy_score(y_true, y_pred))
        
