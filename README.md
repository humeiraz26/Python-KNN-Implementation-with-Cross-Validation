# Python-KNN-Implementation-with-Cross-Validation
 Developed custom KNN classifiers in Python for diverse datasets including Hayes-Roth, Car Evaluation, and Breast Cancer data.
# Report

**Humeira (1002037944)**

## Hayes Roth Dataset

The `KNN` class is defined in the code, and it has two methods: `fit()`, which trains the classifier on the data, and `predict()`, which forecasts the class of a new instance. The mean and standard deviation for each feature for each class are calculated using the `fit()` function. The greatest likelihood class is returned by the `predict()` method, which computes the likelihoods of each new instance for each class using a normal probability density function.

The `k_fold_cross_validation()` function uses the KNN classifier to carry out k-fold cross-validation. The dataset is divided into k folds, the classifier is trained on k-1 folds, and the classifier is tested on the final fold. It does this k times, computing the classifier's accuracy after each fold. Along with the mean accuracy of the classifier, it produces a list of accuracy values for each fold. The percentage of cases that were correctly categorized is known as accuracy. The function also outputs the mean accuracy as well as the accuracy of each fold.

The collection includes nominal features like "name", "hobby", "educational level", together with "marital status".

## Car Evaluation Dataset

The function converts the dataset's category variables to numerical variables after it has been loaded. The dataset is then divided into labels (X and y, respectively) and features. The methods `fit()` and `predict()` of the `KNN` class are defined. The `fit()` function uses the training set's (X and y) values to compute the KNN algorithm's prediction parameters. The class labels for each instance in the test set are predicted by the `predict()` function using the test set (X) and the computed parameters. 

After that, the k-fold cross-validation function is defined to operate on the dataset. The dataset is divided into k folds; k is a user-defined parameter with a default value of 10. The KNN model is then trained on k-1 folds, and its performance is assessed on the final fold. In order to use each fold as a test set once, this operation is done k times. The function provides the mean accuracy over all folds as well as the accuracy for each fold.

## Breast Cancer Dataset

In the code, we do k-fold cross-validation with a KNN classifier after pre-processing the data by transforming category variables to numerical variables. The k-fold cross-validation function and the KNN classifier code are given. Using the replace method of the pandas DataFrame, preprocessing was completed by replacing all "?" values with NaN (not a number) values in the dataset. Then it uses the dropna function to remove any rows that have NaN values. It is simple to guarantee that the dataset is clean and suitable for use in the construction of the KNN classifier by removing the rows having NaN values. 

The data is divided into k folds for the k-fold cross-validation, which runs training and testing k times. One of the folds is utilized as the test set and the remaining k-1 folds are used for training in each iteration. By contrasting the predicted labels with the actual labels in the test set, the accuracy of the model is calculated.

## Conclusion

On the Hayes-Roth, Car Evaluation, and Breast Cancer datasets, the KNN method built in Python from scratch achieved a custom mean accuracy of 54.50%, 79.04%, and 73.87%, respectively. On the same datasets, using the scikit-learn implementation, there are accuracy rates of 54.615%, 88.539%, and 75.80%, respectively. To summarize, a complete Python implementation of the KNN method was accomplished utilizing k-fold cross-validation on three datasets.

## References

- [Tutorial to Implement K-Nearest Neighbors in Python from Scratch](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)
- [K-Fold Cross-Validation](https://machinelearningmastery.com/k-fold-cross-validation/)
- [Naive Bayes Classifier from Scratch in Python](https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/)
- [K-Fold Cross-Validation](https://machinelearningmastery.com/k-fold-cross-validation/)
- [ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch)
- [Implementing a Naive Bayes Classifier for Text Categorization in Five Steps](https://towardsdatascience.com/implementing-a-naive-bayes-classifier-for-text-categorization-in-five-steps-f9192cdd54c3)
