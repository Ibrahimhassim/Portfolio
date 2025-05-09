# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK resources for tokenization and lemmatization.
nltk.download('punkt')
nltk.download('wordnet')

# Load 20 Newsgroups dataset
# This loads Loading the 20 Newsgroups dataset, specifically the categories: 'alt.atheism',
# 'soc.religion.christian', 'comp.graphics', 'sci.med'. Fetching both the training and testing
# subsets, excluding headers, footers, and quotes from the text data.
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

# Define a function for lemmatization
# Creating a custom tokenizer class LemmaTokenizer that uses nltk's WordNetLemmatizer to lemmatize
# tokens in a document. The __call__ method tokenizes the input document and returns the lemmatized
# tokens.
class LemmaTokenizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    def __call__(self, doc):
        tokens = word_tokenize(doc)
        return [self.lemmatizer.lemmatize(token) for token in tokens]

# Create pipeline components with TF-IDF, LDA, and Logistic Regression.
# TfidfVectorizer to convert the text data into TF-IDF features, using the custom LemmaTokenizer,
# ignoring English stop words, and setting max_df and min_df to filter terms.
# LatentDirichletAllocation (LDA) to reduce dimensionality and extract topics from the TF-IDF features.
# LogisticRegression to classify the documents based on the topics.
tfidf = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words='english', max_df=0.95, min_df=2)
lda = LatentDirichletAllocation(n_components=5, max_iter=5, learning_method='online', random_state=42)
logreg = LogisticRegression(max_iter=1000)

# Construct pipeline with TF-IDF, LDA, and Logistic Regression
# Creating a pipeline that sequentially applies TfidfVectorizer, LatentDirichletAllocation,
# and LogisticRegression.
pipeline = Pipeline([
    ('tfidf', tfidf),
    ('lda', lda),
    ('classifier', logreg)
])

# Split data into training and testing sets.
X_train, X_test = newsgroups_train.data, newsgroups_test.data
y_train, y_test = newsgroups_train.target, newsgroups_test.target

# Fit pipeline
# Fitting the pipeline to the training data. The fit method trains the TF-IDF vectorizer,
# LDA model, and logistic regression model sequentially.
print("Fitting pipeline...")
pipeline.fit(X_train, y_train)
print("Pipeline fitted successfully.")

# Predictions on test data
# Making predictions on the test data using the fitted pipeline.
print("Making predictions...")
y_pred = pipeline.predict(X_test)
print("Predictions made successfully.")

# Evaluate the model
# Calculating and printing the accuracy of the model on the test data.
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Classification report
# Printing the classification report, which includes precision, recall, and F1-score for each
# category.
print(classification_report(y_test, y_pred, target_names=newsgroups_test.target_names))

# Confusion matrix
# Generating and plotting the confusion matrix to visualize the performance of the model in terms
# of true positive, false positive, true negative, and false negative predictions for each category.
# The plot is displayed with the category names as labels on both axes.
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(newsgroups_test.target_names))
plt.xticks(tick_marks, newsgroups_test.target_names, rotation=45)
plt.yticks(tick_marks, newsgroups_test.target_names)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')

plt.tight_layout()
plt.show()

# The question that this model is answering -
# Given a document, can we accurately classify it into one of the specified categories
# ('alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med')?"
#
# What is text clasification -
# According to (Amazon Web Services, Inc., n.d.) text classification is a machine learning
# technique used to categorize text into predefined labels. This involves organizing, structuring,
# and labeling text data for various applications such as sentiment analysis, content moderation,
# document management, and customer support. By leveraging natural language processing and various
# algorithms, text classification improves accuracy, enables real-time analytics,
# and scales effectively to handle large volumes of multilingual data.

# What is Lemmatization? -
# According to (Saumyab271, 2022) Lemmatization is a process in natural language processing that
# reduces words to their base or root form, typically aiming to normalize variants of words.
# Unlike stemming, which chops off prefixes or suffixes without considering the context,
# lemmatization ensures that the resulting word belongs to the language and is a meaningful
# word that can be found in a dictionary. For example, lemmatization would convert
# "running" to "run" and "better" to "good", maintaining the semantic meaning of words across
# different forms.

# What is Tokenization ? -
# According to (SearchSecurity, n.d.) Tokenization is a method used in data security where
# sensitive information is substituted with unique identification symbols that retain the
# necessary data elements without compromising its security. This technique aims to reduce the
# amount of sensitive data stored by replacing it with surrogate, non-sensitive information,
# thereby enhancing security in transactions like credit card payments and e-commerce.
# Tokenization helps businesses comply with industry regulations and safeguard sensitive data
# such as financial transactions, medical records, and personal information in various systems.

# What is TF-IDF? -
# According to (MonkeyLearn Blog, 2019) TF-IDF (term frequency-inverse document frequency) is a
# statistical measure used to assess the importance of a word within a document collection.
#
# It calculates relevance by multiplying two values: how frequently a word appears in a document
# (term frequency), and the inverse document frequency of the word across all documents in the
# collection.
#
# This method finds extensive application in automated text analysis and plays a crucial role in
# scoring words for Natural Language Processing (NLP) tasks in machine learning.
#
# Originally developed for document search and information retrieval, TF-IDF assigns higher scores
# to words that appear frequently within a document but are relatively rare across all documents.
# Common words like "this," "what," and "if," which appear in many documents, receive lower scores
# because they are less meaningful in distinguishing one document from another.

# What is Latent Dirichlet Allocation? -
# According to (ibm.com, 2024) Latent Dirichlet Allocation (LDA) is a technique used for topic
# modeling, which identifies the main topics and their distribution across a set of documents.
#
# LDA, distinct from linear discriminant analysis in machine learning, employs a Bayesian approach
# to topic modeling. It operates by probabilistically assigning topics to each document based on
# the distribution of words.
#
# Topic modeling, a natural language processing (NLP) method, applies unsupervised learning to
# large text datasets. Its goal is to generate a concise set of terms that summarize the predominant
# themes within the documents. Through this process, topic models uncover latent topics or
# overarching themes that characterize the entire document collection.

# What are Piplines ? -
# According to (ibm.com, 2024) A data pipeline in data analytics is a structured method for
# gathering raw data from various sources, transforming it, and storing it in a central location
# like a data lake or data warehouse for analysis.
#
# Before data is stored, it goes through preprocessing steps such as filtering and aggregating to
# ensure consistency, especially in relational databases where data structure alignment is crucial.
#
# Data pipelines facilitate data science and business intelligence by handling data from sources
# like APIs, databases, and files, preparing it for analysis through steps like data cleansing
# and integration. This process tracks data lineage, documenting where and how data is stored
# across different systems.

# Why is the Dataset appropriate for text processing? -
# The 20 Newsgroups dataset from sklearn is highly appropriate for text processing tasks for
# several reasons:
# Diverse Categories: It contains approximately 20,000 newsgroup documents spread across
# 20 different categories, providing a broad spectrum of topics for robust training and evaluation.
#
# Standard Benchmark: Widely used in the machine learning community, it serves as a standard
# benchmark for evaluating text classification algorithms, facilitating comparison with other
# models and techniques.
#
# Preprocessed Data: Sklearn offers a version that is already preprocessed, which simplifies the
# process of transforming raw text into usable features for machine learning models.
#
# Versatility: Suitable for a variety of tasks beyond text classification, including topic
# modeling, clustering, and feature extraction, making it a flexible resource for text analytics
# experiments.

# What analysis is performed on the dataset -
# Data Loading: The 20 Newsgroups dataset was loaded, specifically focusing on the categories
# 'alt.atheism', 'soc.religion.christian', 'comp.graphics', and 'sci.med'. Both training and
# testing subsets were fetched, and certain textual components like headers, footers, and quotes
# were removed from the data.
#
# Text Preprocessing:
#
# Tokenization: Each document in the dataset was tokenized using NLTK's word_tokenize function,
# which splits the text into individual words or tokens.
# Lemmatization: A custom tokenizer class LemmaTokenizer was created to lemmatize tokens using
# NLTK's WordNetLemmatizer. Lemmatization reduces words to their base or root form
# (e.g., 'running' to 'run'), which can improve the accuracy of text analysis by consolidating
# variant forms of words.
#
# Pipeline Setup:
#
# TF-IDF Vectorization: The TfidfVectorizer was employed to convert the text data into TF-IDF
# (Term Frequency-Inverse Document Frequency) features. This step assigns weights to words based
# on their frequency in the document and their rarity across all documents.
# Latent Dirichlet Allocation (LDA): Used to perform topic modeling on the TF-IDF features.
# LDA helps in identifying latent topics within the documents and reduces the dimensionality of
# the feature space.
# Logistic Regression: The final step in the pipeline is Logistic Regression, which is used to
# classify the documents into one of the specified categories based on the topics identified by LDA.
#
# Model Training and Evaluation:
#
# Pipeline Fitting: The pipeline (TF-IDF Vectorizer -> LDA -> Logistic Regression) was fitted to
# the training data. This process trains all components of the pipeline sequentially on the
# training dataset.
# Prediction: After fitting the pipeline, predictions were made on the test data using the
# trained model.
# Evaluation Metrics: Various evaluation metrics were computed to assess the performance of the
# classification model, including accuracy, precision, recall, and F1-score. Additionally, a
# confusion matrix was generated and plotted to visualize the model's performance in terms of
# true positives, false positives, true negatives, and false negatives for each category.
# Overall, the analysis involved preparing the text data, transforming it into numerical features
# suitable for machine learning, training a classification model, and evaluating its performance
# on unseen data from the test set. This comprehensive approach ensures that the model is capable
# of accurately classifying new documents into the specified categories.
#
# The confusion matrix graph -
# Axes Labels:
#
# The y-axis (vertical) represents the true labels (actual categories from the test set).
# The x-axis (horizontal) represents the predicted labels (categories predicted by the model).
# Class Labels:
#
# The matrix shows labels for four categories: alt.atheism, comp.graphics, sci.med, and soc.religion.
# christian.
# Color Intensity:
#
# The color intensity indicates the number of instances. Darker colors represent higher numbers.
# Diagonal Cells:
#
# The cells along the diagonal (from the top-left to the bottom-right) represent the number of
# correct predictions for each category. These are the true positives.
# For example, if the cell at (1,1) for comp.graphics is dark, it indicates a high number of
# correct predictions for comp.graphics.
# Off-Diagonal Cells:
#
# The off-diagonal cells represent misclassifications. The position of the cell indicates the true
# label and the predicted label.
# For example, if the cell at (1,3) is dark, it means many comp.graphics documents were incorrectly
# classified as soc.religion.christian.
# Observations from the Confusion Matrix
# Overall Accuracy:
#
# Ideally, a perfectly accurate classifier will have all non-diagonal cells as zero (lightest color),
# and all diagonal cells will be the darkest.
# In this matrix, we see that the diagonal cells are not very dark, indicating that the model has
# not performed well in correctly classifying the documents.
# Misclassification Patterns:
#
# The darkest cell appears to be at the intersection of soc.religion.christian for both true and
# predicted labels. This suggests that documents from this category were predicted correctly more
# often than others.
# There is significant misclassification between categories, as indicated by darker off-diagonal
# cells.
# Specific Misclassifications:
#
# alt.atheism documents seem to have been frequently misclassified into other categories, as the
# row for alt.atheism has darker cells off the diagonal.
# comp.graphics and sci.med also show misclassifications, with notable confusion between these
# categories and others.
# Conclusion
# The confusion matrix reveals that:
#
# The classifier has a higher success rate in predicting soc.religion.christian compared to other
# categories.
# There is considerable confusion between the categories alt.atheism, comp.graphics, and sci.med.
# The overall performance of the classifier needs improvement, as evidenced by the lack of darker
# diagonal cells and the presence of many darker off-diagonal cells.
#


# Code bibliography
# https://www.youtube.com/watch?v=LQQbW3Pve5U
# https://www.youtube.com/watch?v=GsO5fsxlQ3g
# https://www.youtube.com/watch?v=ZNze2VHgsrE
# https://www.youtube.com/watch?v=e5W2ByNuDIE
# https://www.youtube.com/watch?v=C3MQsVX6xzk
# scikit-learn.org. (n.d.). The 20 newsgroups text dataset — scikit-learn 0.15-git documentation.
# [online] Available at: https://scikit-learn.org/0.15/datasets/twenty_newsgroups.html
# [Accessed 20 Jun. 2024].
# https://www.youtube.com/watch?v=12DpJ4MVork
# https://www.youtube.com/watch?v=2WFg2JCaMuc
# https://www.youtube.com/watch?v=uoHVztKY6S4
# https://www.youtube.com/watch?v=nNvPvvuPnGs
# https://www.youtube.com/watch?v=ALQ88I6yNRE
# https://www.youtube.com/watch?v=D2V1okCEsiE
# https://www.youtube.com/watch?v=JthgChPKFG4
# https://www.youtube.com/watch?v=Kdsp6soqA7o
# Rink, K. (2022). Advanced Pipelines with scikit-learn. [online] Medium.
# Available at: https://towardsdatascience.com/advanced-pipelines-with-scikit-learn-4204bb71019b.
# Müller, A.C. and Guido, S. (2017). Introduction to machine learning with
# Python : a guide for data scientists. Beijing: O’reilly.

# Theory Bibligraphy
# MonkeyLearn (2018). Text Classification: [online] MonkeyLearn. Available
# at: https://monkeylearn.com/text-classification/.
# Amazon Web Services, Inc. (n.d.). What is Text Classification? - Text Classification Explained -
# AWS. [online] Available at: https://aws.amazon.com/what-is/text-classification/.
# www.ibm.com. (2024). What Is a Data Pipeline? | IBM. [online] Available
# at: https://www.ibm.com/topics/data-pipeline#:~:text=Contributor%3A%20Cole%20Stryker-
# [Accessed 21 Jun. 2024].
# Scikit learn (2018). sklearn.feature_extraction.text.TfidfVectorizer — scikit-learn 0.20.3
# documentation. [online] Scikit-learn.org. Available
# at: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html.
# Saumyab271 (2022). Stemming vs Lemmatization in NLP: Must-Know Differences.
# [online] Analytics Vidhya. Available at: https://www.analyticsvidhya.com/blog/2022/06/stemming-vs-lemmatization-in-nlp-must-know-differences/#:~:text=Lemmatization%20is%20a%20linguistic%20process.
# SearchSecurity. (n.d.). What is Tokenization? [online] Available
# at: https://www.techtarget.com/searchsecurity/definition/tokenization.
# MonkeyLearn Blog. (2019). Understanding TF-ID: A Simple Introduction.
# [online] Available at: https://monkeylearn.com/blog/what-is-tf-idf/#:~:text=TF%2DIDF%20(term%20frequency%2D.
# www.ibm.com. (2024). What is Latent Dirichlet allocation | IBM. [online] Available
# at: https://www.ibm.com/topics/latent-dirichlet-allocation#:~:text=Latent%20Dirichlet%20allocation%20is%20a [Accessed 21 Jun. 2024].
#
