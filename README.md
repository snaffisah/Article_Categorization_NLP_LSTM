# Article_categorization_NLP_LSTM
NLP is a Natural Language Processing which is used to analyse a text data. In this analysis, NLP were used with a deep learning model with LSTM neural network approach on an article to categorize it into its category.

### Description
Objective: Create a classifier model to identify category for an article using deep learning

* Model training - Deep learning
* Method: Sequential, LSTM
* Module: Sklearn & Tensorflow

In this analysis, dataset used from https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv

### About The Dataset:
There are 2,225 entries of text article in the dataset which categorized into 5 category:
- Sport
- Tech
- Business
- Entertainment
- Politics

99 duplicated data were found during the analysis and its were removed before seperating the dataset. 

Text data will be used as our feature and category data will be the target label. Before start the training, HTML tags need to be removed from the text and must be in lower cases. After that, words were splitted into elements in array.

For category data, one hot encoder is used to convert it into a format that can be used for the training.

### Deep learning model with LSTM layer
A sequential model was created with 1 LSTM layer and 2 dense layer:
<p align="center">
  <img width="440" src="https://github.com/snaffisah/Article_Categorization_NLP_LSTM/blob/main/Image/sequential%20model%20LSTM.JPG">
</p>

<p align="center">
  <img src="https://github.com/snaffisah/Article_Categorization_NLP_LSTM/blob/main/Image/model%20workflow.JPG">
</p>

Data were trained with 10 epoch:
<p align="center">
  <img src="https://github.com/snaffisah/Article_Categorization_NLP_LSTM/blob/main/Image/epoch.JPG">
</p>

<p align="center">
  <img src="https://github.com/snaffisah/Article_Categorization_NLP_LSTM/blob/main/Image/tensorboard%20graph.JPG">
</p>

The classification report, confiusion matrix and accuracy score achieve as below:
<p align="center">
  <img src="https://github.com/snaffisah/Article_Categorization_NLP_LSTM/blob/main/Image/analysis%20report.JPG">
</p>

### Result
By using the created model, a new article was tested and the category was assigned correctly.
<p align="center">
  <img "https://github.com/snaffisah/Article_Categorization_NLP_LSTM/blob/main/Image/prediction.JPG">
</p>
### How to run the pythons file:
1. Load the module 1st by running 'article_categorization_module.py'
2. Run training file 'article_categorization_train.py' (this step can be skipped)
3. Run 'article_categorization_deploy.py' to test the new article and check the output

Enjoy!
