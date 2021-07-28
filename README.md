# Machine Learning : Resampling and ensemble Learning
# Risky Business
 
![Credit Risk](Images/credit-risk.jpg)

## Background

Mortgages, student and auto loans, and debt consolidation are just a few examples of credit and loans that people seek online. Peer-to-peer lending services such as Loans Canada and Mogo let investors loan people money without using a bank. However, because investors always want to mitigate risk, a client has asked me to help them predict credit risk with machine learning techniques.

In this Tool I have built and evaluated several machine learning models to predict credit risk using data typically see from peer-to-peer lending services. Credit risk is an inherently imbalanced classification problem (the number of good loans is much larger than the number of at-risk loans), so I have employed different techniques for training and evaluating models with imbalanced classes. I have used the imbalanced-learn and Scikit-learn libraries to build and evaluate models using the two following techniques:

1. [Resampling](#Resampling)
2. [Ensemble Learning](#Ensemble-Learning)

- - -

## Files

[Resampling Starter Notebook](credit_risk_resampling.ipynb)

[Ensemble Starter Notebook](credit_risk_ensemble.ipynb)

[Lending Club Loans Data](LoanStats_2019Q1.csv.zip)

- - -

### Resampling

Using the [imbalanced learn](https://imbalanced-learn.readthedocs.io) library I have resampled the LendingClub data and built and evaluate logistic regression classifiers using the resampled data.

To begin the following steps were conducted:

1. Data was read from the CSV into a DataFrame.

2. Split the data into Training and Testing sets.

3. Scaled the training and testing data using the `StandardScaler` from `sklearn.preprocessing`.

4. Ran a Simple Logistic Regression and performed the following:
    * Fit the `logistic regression classifier`.
    * Calculate the `balanced accuracy score`.
    * Display the `confusion matrix`.
    * Print the `imbalanced classification report`.

Next :

1. Oversampled the data using the `Naive Random Oversampler` and `SMOTE` algorithms.

2. Undersampled the data using the `Cluster Centroids` algorithm.

3. Over- and undersampled using a combination `SMOTEENN` algorithm.


For each of the above the follwing actions were taken :

1. Trained a `logistic regression classifier` from `sklearn.linear_model` using the resampled data.

2. Calculated the `balanced accuracy score` from `sklearn.metrics`.

3. Printed the `confusion matrix` from `sklearn.metrics`.

4. Printed the `imbalanced classification report` from `imblearn.metrics`.


The following conclutions were drawn:

* Which model had the best balanced accuracy score?

  SMOTE oversampling, ClusterCentroids undersampling and SMOTEENN combined over and under sampling have the highest Balanced Accuracy Score at 99.5%.
  
* Which model had the best recall score?

  All the models SimpleLogisticRegression, NaiveRandomOversampling, SMOTE oversampling, ClusterCentroids undersampling, and SMOTEENN combined over & under all of them thave the same Recall value at 99%
 
* Which model had the best geometric mean score?

  NaiveRamdom oversampling, SMOTE oversampling, and Combined over&under sampling SMOTEENN model have the higest geometic mean score of 100%.

### Ensemble Learning 

In this Tool, I have trained and compared two different ensemble classifiers to predict loan risk and evaluated each model. The following two modles were used  [Balanced Random Forest Classifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html) and the [Easy Ensemble Classifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html). 

To begin, the following steps were performed:

1. The data was read from the csv file into a DataFrame.

2. The Data was Split into training and testing sets.

3. Scaled the training and testing data using the `StandardScaler` from `sklearn.preprocessing`.


Then, the following steps were completed for each model:

1. Trained the model using the quarterly data from LendingClub provided in the `Resource` folder.

2. Calculated the balanced accuracy score from `sklearn.metrics`.

3. Displayed the confusion matrix from `sklearn.metrics`.

4. Generated a classification report using the `imbalanced_classification_report` from imbalanced learn.

5. For the balanced random forest classifier, printed the feature importance sorted in descending order (most important feature to least important) along with the feature score.


The following conclusions were drawn:

* Which model had the best balanced accuracy score?
  EasyEnsembleClassifier has the best balanced accuracy score of 93% compared to the BalancedRandomForest Classifier at 78%. 

* Which model had the best recall score?
  EasyEnsembleClassifier has the best recall score at 94% where as the BalancedRandomForestClassifier is at 87%.

* Which model had the best geometric mean score?
  EasyEnsembleClassifier has the best geometric mean score at 93%, where as the BalancedRandomForestClassifier has a score of 78%.

* What are the top three features?
  he top three features are 'total_rec_prncp' ,'total_pymnt', 'total_pymnt_inv'.

- - -

### References

Refer to the [imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/) and [scikit-learn](https://scikit-learn.org/stable/) official documentation for help with training the models. 

For the ensemble learners, used 100 estimators for both models.
