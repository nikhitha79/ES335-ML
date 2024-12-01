from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.base import DecisionTree  # Assuming you have a DecisionTree class
from metrics import *  # Assuming you have implemented metrics like accuracy, precision, recall

# Generate dataset
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, 
    random_state=1, n_clusters_per_class=2, class_sep=0.5
)
# Visualize the dataset
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# Convert to pandas DataFrame and Series
X = pd.DataFrame(X, columns=['feature1', 'feature2'])
y = pd.Series(y, name='label')


# Split the dataset into training (70%) and testing (30%) sets
split_idx = int(0.7 * len(X))
X_train, X_test, y_train, y_test = X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

# Create and train your DecisionTree model

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)
    tree.fit(X_train, y_train)
    # Make predictions on the test data
    y_hat = tree.predict(X_test)
    #tree.plot()
    print("\nCriteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y_test))
    # Evaluate the performance of the model using your custom metrics
    for cls in y.unique():
        print("Precision: ", precision(y_hat, y_test, cls))
        print("Recall: ", recall(y_hat, y_test, cls))


def cross_validation(X, y, k, depths_list):
    # Initialize variables to track the best results
    best_fold = 0
    best_depth = 0
    max_accuracy = 0

    # Calculate the size of each fold
    fold_size = len(X) // k

    # Perform k-fold cross-validation
    for i in range(k):
        # Split the data into training and test sets
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        test_set = X[test_start:test_end]
        test_labels = y[test_start:test_end]

        training_set = pd.concat([X[:test_start], X[test_end:]], axis=0)
        training_labels = pd.concat([y[:test_start], y[test_end:]], axis=0)

        # Iterate over depths and evaluate model performance
        for depth in depths_list:
            # Train the model
            dt_classifier = DecisionTree(max_depth=depth)
            dt_classifier.fit(training_set, training_labels)

            # Make predictions on the validation set
            fold_predictions = dt_classifier.predict(test_set)

            # Calculate the accuracy of the fold
            fold_accuracy = accuracy(fold_predictions, test_labels)

            # Check if the current depth has higher accuracy
            if fold_accuracy > max_accuracy:
                max_accuracy = fold_accuracy
                best_fold = i + 1
                best_depth = depth

            # Print accuracy and depth for each fold
            print("Fold_region {}: Depth: {} - Accuracy: {:.4f}".format(i + 1, depth, fold_accuracy))

    # Print the optimal results
    print("\nOptimal Fold_region : {} (Depth: {}, Accuracy: {:.4f})".format(best_fold, best_depth, max_accuracy))
    print("\n")
# Example usage
cross_validation(X, y, 5, [3, 4, 5, 6])




print("### NESTED CROSS VALIDATION ###\n")
def nested_cross_validation(X, y, outer_k, inner_k, depth_values):
    outer_fold_size = len(X) // outer_k
    opt_depth = 0
    opt_accuracy = 0
    # Outer loop for cross-validation
    for i in range(outer_k):
        outer_test_start = i * outer_fold_size
        outer_test_end = (i + 1) * outer_fold_size
        outer_test_set = X.iloc[outer_test_start:outer_test_end]
        outer_test_labels = y.iloc[outer_test_start:outer_test_end]

        outer_training_set = pd.concat([X.iloc[:outer_test_start], X.iloc[outer_test_end:]], axis=0)
        outer_training_labels = pd.concat([y.iloc[:outer_test_start], y.iloc[outer_test_end:]], axis=0)

        # Inner loop for hyperparameter tuning
        inner_fold_size = len(outer_training_set) // inner_k
        inner_fold = 0

        # Dictionary to store average performance metrics for each depth
        avg_metrics = {}

        for inner_start in range(0, len(outer_training_set), inner_fold_size):
            inner_end = inner_start + inner_fold_size
            inner_validation_set = outer_training_set.iloc[inner_start:inner_end]
            inner_validation_labels = outer_training_labels.iloc[inner_start:inner_end]

            inner_training_set = pd.concat([outer_training_set.iloc[:inner_start], outer_training_set.iloc[inner_end:]], axis=0)
            inner_training_labels = pd.concat([outer_training_labels.iloc[:inner_start], outer_training_labels.iloc[inner_end:]], axis=0)

            # Iterate over depth values and evaluate model performance
            for depth in depth_values:
                # Train the model
                dt_classifier = DecisionTree(max_depth=depth)
                dt_classifier.fit(inner_training_set, inner_training_labels)

                # Make predictions on the validation set
                fold_predictions = dt_classifier.predict(inner_validation_set)

                # Calculate the accuracy of the fold
                fold_accuracy = accuracy(fold_predictions, inner_validation_labels)

                # Update the average metrics dictionary
                avg_metrics.setdefault(depth, 0)
                avg_metrics[depth] += fold_accuracy

            inner_fold += 1

        # Calculate average metrics for the current outer fold
        for depth, acc in avg_metrics.items():
            avg_metrics[depth] /= inner_fold

        # Find the depth with the highest average accuracy
        best_depth = max(avg_metrics, key=avg_metrics.get)

        # Train the final model with the best depth on the outer training set
        final_model = DecisionTree(max_depth=best_depth)
        final_model.fit(outer_training_set, outer_training_labels)

        # Make predictions on the outer test set
        fold_predictions = final_model.predict(outer_test_set)

        # Calculate and print the performance metrics for the fold
        fold_accuracy = accuracy(fold_predictions, outer_test_labels)
        print(f"Outer Fold {i + 1}: Best Depth: {best_depth} - Accuracy: {fold_accuracy}")
        if fold_accuracy>opt_accuracy:
            opt_accuracy = fold_accuracy
            opt_depth = best_depth
    print("Optimal depth = ", opt_depth)

# Example usage
depth_values = [3, 4, 5, 6]
nested_cross_validation(X, y, outer_k=5, inner_k=3, depth_values=depth_values)