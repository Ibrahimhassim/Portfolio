# Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Load the disease dataset from the CSV file
disease_data = pd.read_csv('C:/Users/ibrah/Downloads/Disease_Symptom_and_patient_profile_dataset.csv')

# Perform one-hot encoding on categorical variables by using the get dumies function from pandas,
# It creates binary columns for each category of categorical variables specified in the
# columns parameter.
disease_data_encoded = pd.get_dummies(disease_data, columns=['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender'])

# Replace categorical values with numeric labels using the replace function for these two columns.
disease_data_encoded['Blood Pressure'] = disease_data_encoded['Blood Pressure'].replace({'Low': 1, 'Normal': 2, 'High': 3})
disease_data_encoded['Cholesterol Level'] = disease_data_encoded['Cholesterol Level'].replace({'Low': 1, 'Normal': 2, 'High': 3})

# Drop irrelevant columns in this case it was disease as it was not needed.
columns_to_drop = ['Disease']
disease_data_encoded = disease_data_encoded.drop(columns_to_drop, axis=1)

# Separate features and target variable,from the DataFrame. Here, 'X' contains the features,
# obtained by dropping the 'Outcome Variable' column, and 'y' contains the target variable.
X = disease_data_encoded.drop(['Outcome Variable'], axis=1)
y = disease_data_encoded['Outcome Variable']

# Convert 'Negative' and 'Positive' labels to binary format (0 and 1),
y_binary = y.replace({'Negative': 0, 'Positive': 1})

# Split the data into training and testing sets (80% training, 20% testing),
# from scikit-learn It splits 'X' and 'y_binary' into training and testing sets with an 80-20 ratio,
# where 80% of the data is used for training and 20% for testing.
# The random_state parameter ensures reproducibility of the split.
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Initialize and train the decision tree classifier,  and fits it to the training data
# using the fit method. This step trains the classifier on the training dataset.
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Next, train the Naïve Bayes Classifier and the K-NN classifier,
# evaluate their accuracy, and compare their performance:

# Initialize and train the Naïve Bayes classifier using the Gaussian Naive Bayes implementation
# and trains it on the training data.
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# Initialize and train the K-NN classifier with 5 neighbors and trains it on the training data.
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Makes predictions on the test set for each classifier.
# For the decision tree and Naïve Bayes classifiers, it uses the predict method,
# while for the K-NN classifier, it predicts probabilities using the predict_proba method
# and selects the probability of the positive class.

# Make predictions on the test set for decision tree
y_pred_dt = decision_tree.predict(X_test)

# Make predictions on the test set for Naïve Bayes
y_pred_nb = naive_bayes.predict(X_test)

# Make predictions on the test set for K-NN
y_prob_knn = knn_classifier.predict_proba(X_test)[:, 1]

# Calculates the accuracy score for each classifier using the accuracy_score function from
# scikit-learn. It compares the predicted labels with the actual labels in the test set
# and prints the accuracy scores.

# Calculate accuracy for decision tree
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)

# Calculate accuracy for Naïve Bayes
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print("Naïve Bayes Accuracy:", accuracy_nb)

# Calculate accuracy for K-NN
accuracy_knn = accuracy_score(y_test, y_prob_knn.round())
print("K-NN Accuracy:", accuracy_knn)

# Calculates the Receiver Operating Characteristic (ROC) curve for Naïve Bayes and
# K-NN classifiers using the roc_curve function from scikit-learn.
# It then plots the ROC curves using matplotlib, along with the diagonal line representing
# a random classifier. Finally, it sets plot limits, labels, title, and legend before displaying
# the plot.

# Plotting the ROC curve for both classifiers:
# Calculate the ROC curve for Naïve Bayes
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_pred_nb)

# Calculate the ROC curve for K-NN
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

# Plot ROC curve for Naïve Bayes
plt.figure()
plt.plot(fpr_nb, tpr_nb, color='darkorange', lw=2, label='Naïve Bayes ROC curve')

# Plot ROC curve for K-NN
plt.plot(fpr_knn, tpr_knn, color='green', lw=2, label='K-NN ROC curve (area = %0.2f)' % roc_auc_knn)

# Plot the diagonal line (random classifier)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Set plot limits and labels
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Classification Analysis Definition
# According to (Indicative, n.d.), Classification Analysis is a type of data analysis method that
# which gives categories to a collection of data to allow it to have accurate analysis, this method
# is commonly used in mathematical type analysis such as Linear programming and decision trees.

# Analysis of Graph -
# The ROC curve shows the performance of two different classifiers: a Naive Bayes classifier
# and a K-Nearest Neighbors (K-NN) classifier. The area under the ROC curve (AUC) is a measure
# of how well a classifier performs. A higher AUC indicates a better classifier. In the image,
# the AUC for the K-NN classifier is 0.56, which is not very good. An AUC of 1 would be a perfect
# classifier, and an AUC of 0.5 is no better than random guessing.
# •	The K-NN classifier has an AUC of 0.56, which is not very good. This means that the classifier
# is not much better than random guessing at distinguishing between positive and negative cases.
# •	The ROC curve for the K-NN classifier is closer to the diagonal line than a perfect classifier
# would be.
# This means that the classifier is making a lot of false positive errors.
# Simply put the ROC curve in the image suggests that both classifiers are not performing very well.
# You could try to improve the performance of the classifiers by collecting more data, training the
# classifiers on a different set of features, or using a different classification algorithm.

# Why the dataset is appropriate -
# Definition – According to (sciencedirect.com, n.d.), categorical data is made up of non-numeric
# data where objects or items are given categories without any specific order.
# This dataset is about disease symptoms and patient profile data and is made up of ten columns most
# of which are nominal categorical data except for age which is numerical data column.
# The other nine columns consists disease name which has a list of all the various disease,
# the next six columns are symptoms, such as Fever -which is a yes or no column indicating if
# a person has a Fever or not, Cough – which indicates if the person is coughing or not,
# Fatigue - which indicates if the person is experiencing any fatigue or not,
# Difficulty breathing - which indicates if the person is experiencing any difficulty breathing
# or not, Blood Pressure - which indicates what the person’s blood pressure is either low, normal,
# or high, Cholesterol Level - which indicates what the persons cholesterol levels are, either normal,
# low, or high, the last two are Gender – which just indicates if the person is male female,
# and outcome variable which indicates if the person is either negative or positive.
# All these including age make up the categorical dataset.
# This dataset is appropriate because of a few reasons such as,
#
# 1.	Categorical Data: Disease datasets often contain categorical variables like symptoms, causes,
# and types of diseases, which are conducive to classification tasks. Decision trees are adept
# at handling categorical data as they can split the data based on these categories.
# 2.    According to (Christoph Molnar, 2019),	Interpretable Results:
# Decision trees provide a clear and interpretable decision-making
# process. They segment the data into hierarchical structures of if-then rules, which can mimic the
# diagnostic process used by medical professionals.
# 3.    According to (Krisalay, 2024),	Feature Importance:
# Decision trees can also provide insights into the most important features
# for distinguishing between different diseases. This can be valuable for understanding which symptoms
# or factors play a significant role in diagnosing a particular disease.
# 4.	According to (linkedin.com, n.d.),	Handling Nonlinear Relationships:
# Decision trees can capture nonlinear relationships between
# features and target variables. In medical datasets, relationships between symptoms and diseases may
# not always be linear, making decision trees a good choice for modelling such data.
# 5.	Handling Missing Values: Disease datasets often contain missing values, which decision trees
# can handle without requiring imputation methods. Decision trees can simply create separate branches
# for instances with missing values, ensuring that no data is discarded during the analysis.
# 6.	According to (AlmaBetter, n.d.),Ensemble Methods:
# Decision trees can also be used as base learners in ensemble methods like
# random forests or gradient boosting, which can further enhance the predictive performance of the
# model by combining multiple decision trees.
#
# Overall, the disease well-suited for classification analysis using decision trees due to its
# categorical nature, interpretability, ability to handle nonlinear relationships, and capacity
# to handle missing data.

# The analysis that was performed -
# Classification analysis is performed.
# 1.	Data Preprocessing:
# •	One-hot encoding: Categorical variables like symptoms and gender are converted into binary columns
# using one-hot encoding.
# •	Irrelevant column removal: The column "Disease" is dropped from the dataset as it's not needed for
# classification.
# 2.	Data Splitting:
# •	The dataset is split into features (X) and target variable (y).
# •	The target variable is converted into binary format, replacing "Negative" and "Positive" labels
# with 0s and 1s.
# •	The data is further split into training and testing sets (80% training, 20% testing) using the
# train_test_split function.
# 3.	Model Training:
# •	Three classification models are trained on the training data:
# •	Decision Tree Classifier
# According to (Dobra, 2009) “this is just a decision tree used for classification tasks as any other
# classifier it uses values of attributes/features of the data to make a class label prediction.”
# •	Naïve Bayes Classifier
# According to (Martins, 2023), the Naïve Bayers classifier is “a machine learning classification
# technique based on a probabilistic approach that assumes each class follows a normal distribution.
# It assumes each parameter has an independent capacity of predicting the output variable.
# It can also predict the probability of a dependent variable to be classified in each group.”
# •	K-Nearest Neighbours (K-NN) Classifier
# According to (IBM, 2023), “the KNN algorithm is a non-parametric, supervised learning classifier,
# which makes use of proximity in order to make classifications or predictions about the grouping of
# an individual data point”.
# I would go about creating my models by performing:
# Data Preprocessing
# Feature engineering
# Hyper Parameter tuning (If needed)
# Lastly the model evaluation.
# 4.	Model Evaluation:
# •	Accuracy scores are calculated for each classifier on the testing set using the accuracy_score
# function.
# •	The accuracy scores are printed to compare the performance of different classifiers. This also
# shows how the model was evaluated.
# •	The 3 different accuracy scores were as follows.
# •	Decision Tree Accuracy: 0.7428571428571429 - 74%
# •	Naïve Bayes Accuracy: 0.6142857142857143 - 61%
# •	K-NN Accuracy: 0.5142857142857142 - 51%
#
# 5.	Plotting:
# •	The results were plotted using the matplotlib library in Python. Specifically, the ROC curves were
# plotted using matplotlib.pyplot.plot() function.
# calculate ROC Curves:
# The ROC curves for Naïve Bayes and K-NN classifiers were calculated using the roc_curve function
# from sklearn.metrics. This function returns the false positive rate, true positive rate, and
# thresholds for different classification thresholds.
#
# Plotting ROC Curves:
# The ROC curve for Naïve Bayes classifier was plotted using plt.plot() function, specifying the false
# positive rate (FPR) on the x-axis and true positive rate (TPR) on the y-axis. The color, line width,
# and label were customized for clarity. Similarly, the ROC curve for K-NN classifier was plotted in
# the same way, with different color and label.
#
# Plotting Diagonal Line:
# A diagonal line representing a random classifier was plotted using plt.plot() function.
# This line runs from the bottom-left corner (0,0) to the top-right corner (1,1) of the plot.
#
# Customizing Plot:
# The plot limits, labels, title, and legend were customized using various matplotlib functions like
# plt.xlim(), plt.ylim(), plt.xlabel(), plt.ylabel(), plt.title(), and plt.legend().
#
# Display Plot:
# Finally, the plot was displayed using plt.show().
#
# 6.	Receiver Operating Characteristic (ROC) Curve:
# •	ROC curves are calculated for Naïve Bayes and K-NN classifiers using the roc_curve function.
# •	The area under the ROC curve (AUC) is calculated for the K-NN classifier.
# •	ROC curves are plotted using matplotlib to visualize the performance of these classifiers in
# terms of true positive rate versus false positive rate.
# The analysis aims to evaluate and compare the performance of different classifiers in predicting
# disease outcomes based on patient features and symptoms. It provides insights into which classifier
# performs better and can be more effective for this classification task.
# The analysis of the ROC graph is as follows:
# The ROC curve shows the performance of two different classifiers: a Naive Bayes classifier and
# a K-Nearest Neighbors (K-NN) classifier. The area under the ROC curve (AUC) is a measure of how
# well a classifier performs. A higher AUC indicates a better classifier. In the image,
# the AUC for the K-NN classifier is 0.56, which is not very good. An AUC of 1 would be a perfect
# classifier, and  an AUC of 0.5 is no better than random guessing.

# The Naive Bayes ROC curve being above the diagonal indicates that the classifier is better than
# random guessing. However, the ROC curve is not very far from the diagonal, suggesting that the
# model's performance is relatively poor. It shows that the Naive Bayes classifier has limited
# ability to distinguish between the positive and negative classes in this dataset.

# •	The K-NN classifier has an AUC of 0.56, which is not very good. This means that the classifier
# is not much better than random guessing at distinguishing between positive and negative cases.
# Although the exact AUC for Naive Bayes is not provided in the plot, the shape of the curve suggests
# that it is likely not much higher than that of the K-NN classifier

# •	The ROC curve for the K-NN classifier is closer to the diagonal line than a perfect classifier
# would be. This means that the classifier is making a lot of false positive errors.
# Simply put the ROC curve in the image suggests that both classifiers are not performing very well.
# You could try to improve the performance of the classifiers by collecting more data,
# training the classifiers on a different set of features, or using a different classification
# algorithm.




# Bibliography for the code.
# https://www.google.com/search?sca_esv=450a9a9b983914d8&sxsrf=ACQVn08yvGWK5iaixdb-Uvrj6WX-MGAZFQ:1714567949301&q=how+do+i+perform+one+hot+encoding+on+categorical+variables+in+pycharm&tbm=vid&source=lnms&prmd=visnbmtz&sa=X&ved=2ahUKEwjrqb6sv-yFAxXM_rsIHaJcCpYQ0pQJegQIDRAB&biw=1707&bih=781&dpr=1.13#fpstate=ive&vld=cid:44c7dd67,vid:YOR6rQTTEAQ,st:0
# https://www.google.com/search?sca_esv=450a9a9b983914d8&sxsrf=ACQVn08b51am_pp-uWGCj1XUq1FQ3FmP7w:1714586387137&q=how+do+i+use+a+drop+function+in+python+to+drop+an+unwanted+column+from+my+dataset&tbm=vid&source=lnms&prmd=visnbmtz&sa=X&ved=2ahUKEwi5vqqEhO2FAxUfR_EDHRpqCeQQ0pQJegQIEBAB&biw=1707&bih=781&dpr=1.13#fpstate=ive&vld=cid:627b7078,vid:WuNGsB16Dzo,st:0
# https://www.google.com/search?sca_esv=b511c58466c4623d&sxsrf=ACQVn09eTWACF1lCSUAOYJdkCE2UsilB5Q:1714587178463&q=how+do+i+Initialize+and+train+the+decision+tree+classifier+and+fit+it+to+the+training+data+using+the+fit+method&tbm=vid&source=lnms&prmd=visnbmtz&sa=X&ved=2ahUKEwiGg9X9hu2FAxXTUqQEHfgCBUgQ0pQJegQIDRAB&biw=1707&bih=781&dpr=1.13#fpstate=ive&vld=cid:06c96212,vid:Wjc1sFU7UCY,st:0
# https://www.google.com/search?q=how+do+i+Initialize+and+train+the+K-NN+classifier+with+5+neighbors+and+trains+it+on+the+training+data+using+sk+learn&sca_esv=450a9a9b983914d8&biw=1707&bih=781&tbm=vid&sxsrf=ACQVn08w_uRdTXrVYfuLrvB3O_o7CDMcMg%3A1714586858287&ei=6oQyZuiGEZ2L7NYPjOqw2AI&ved=0ahUKEwjoo__khe2FAxWdBdsEHQw1DCsQ4dUDCA4&uact=5&oq=how+do+i+Initialize+and+train+the+K-NN+classifier+with+5+neighbors+and+trains+it+on+the+training+data+using+sk+learn&gs_lp=Eg1nd3Mtd2l6LXZpZGVvInRob3cgZG8gaSBJbml0aWFsaXplIGFuZCB0cmFpbiB0aGUgSy1OTiBjbGFzc2lmaWVyIHdpdGggNSBuZWlnaGJvcnMgYW5kIHRyYWlucyBpdCBvbiB0aGUgdHJhaW5pbmcgZGF0YSB1c2luZyBzayBsZWFybjIEECMYJzIEECMYJzIEECMYJzIKEAAYgAQYQxiKBTIFEAAYgAQyBRAAGIAEMgUQABiABDIIEAAYgAQYogRI-D5QAFiyPXAAeACQAQCYAf0CoAHiEqoBBTItOC4xuAEDyAEA-AEB-AECmAIJoAKAE8ICCxAAGIAEGJECGIoFwgILEAAYgAQYsQMYgwHCAg4QABiABBixAxiDARiKBcICCBAAGIAEGLEDwgINEAAYgAQYsQMYQxiKBcICCxAAGIAEGLEDGIoFmAMAkgcFMi04LjGgB8Z2&sclient=gws-wiz-video#fpstate=ive&vld=cid:69a6fead,vid:Ze59HBg_AF0,st:0
# https://www.google.com/search?sca_esv=450a9a9b983914d8&sxsrf=ACQVn0-f6UIsE_vjtZE1-v2603-niZsyCw:1714592373201&q=how+to+plot+a+roc+curve+in+python&tbm=vid&source=lnms&prmd=visnbmtz&sa=X&ved=2ahUKEwia6tqqmu2FAxWLh_0HHbdAD6IQ0pQJegQIEhAB&biw=1707&bih=781&dpr=1.13#fpstate=ive&vld=cid:d5a7f8c4,vid:uVJXPPrWRJ0,st:0
# https://www.youtube.com/watch?v=2ynTNY7UmmM
# saturncloud.io. (2023). How to Convert Categorical Data to Numerical Data with Pandas | Saturn Cloud Blog. [online] Available at: https://saturncloud.io/blog/how-to-convert-categorical-data-to-numerical-data-with-pandas/#:~:text=Method%202%3A%20Using%20the%20replace [Accessed 1 May 2024].
# Navlani, A. (2023). Python Decision Tree Classification Tutorial: Scikit-Learn DecisionTreeClassifier. [online] www.datacamp.com. Available at: https://www.datacamp.com/tutorial/decision-tree-classification-python.
# LinkedIn. (n.d.). LinkedIn Login, Sign in. [online] Available at: https://www.linkedin.com/pulse/basics-decision-tree-python-omkar-sutar/.

# Bibliography for the Theory
# www.sciencedirect.com. (n.d.). Categorical Data - an overview | ScienceDirect Topics. [online] Available at: https://www.sciencedirect.com/topics/computer-science/categorical-data
# StudySmarter UK. (n.d.). Decision Tree Method: Applications, Pros & Cons, Examples. [online] Available at: https://www.studysmarter.co.uk/explanations/business-studies/managerial-economics/decision-tree-method/#:~:text=The%20Decision%20Tree%20Method%20comes.
# AlmaBetter. (n.d.). Ensembles of Decision Trees. [online] Available at: https://www.almabetter.com/bytes/tutorials/data-science/ensembles-of-decision-tree.
# Duggal, N. (2022). Advantages of Decision Trees : Everything You Need to Know | Simplilearn. [online] Simplilearn.com. Available at: https://www.simplilearn.com/advantages-of-decision-tree-article.
# www.linkedin.com. (n.d.). How can decision trees improve regression analysis? [online] Available at: https://www.linkedin.com/advice/1/how-can-decision-trees-improve-regression-analysis-bbfsf#:~:text=The%20benefits%20of%20using%20decision [Accessed 2 May 2024].
# Martins, C. (2023). Gaussian Naive Bayes Explained With Scikit-Learn | Built In. [online] builtin.com. Available at: https://builtin.com/artificial-intelligence/gaussian-naive-bayes.
# IBM (2023). What is the k-nearest neighbors algorithm? | IBM. [online] www.ibm.com. Available at: https://www.ibm.com/topics/knn.
# Dobra, A. (2009). Decision Tree Classification. pp.765–769. doi:https://doi.org/10.1007/978-0-387-39940-9_554.
# Sinan Ozdemir (2016). Principles of Data Science learn the techniques an math you need to start making sense of your data. Birmingham Packt December.
# Indicative. (n.d.). What Is Classification Analysis? Data Defined. [online] Available at: https://www.indicative.com/resource/classification-analysis/#:~:text=Classification%20analysis%20is%20a%20data.
# Christoph Molnar (2019). 4.4 Decision Tree | Interpretable Machine Learning. [online] Github.io. Available at: https://christophm.github.io/interpretable-ml-book/tree.html.
# Krisalay (2024). Feature Importance in Decision Tree. [online] Medium. Available at: https://medium.com/@krisalay/feature-importance-in-decision-tree-8e60f2174717#:~:text=Feature%20importance%20analysis%20in%20decision [Accessed 27 May 2024].
# www.linkedin.com. (n.d.). How can you analyze non-linear relationships in datasets? [online] Available at: https://www.linkedin.com/advice/3/how-can-you-analyze-non-linear-relationships-datasets-bqynf#:~:text=%2D%20Decision%20trees%20capture%20non%2Dlinear [Accessed 27 May 2024].
#

# Dataset
# www.kaggle.com. (n.d.). Disease Symptoms and Patient Profile Dataset. [online] Available at: https://www.kaggle.com/datasets/uom190346a/disease-symptoms-and-patient-profile-dataset.


