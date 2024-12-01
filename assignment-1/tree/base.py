# pylint: disable=all

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from tree.utils import information_gain, gini_gain, opt_split_attribute

# Class for the tree node
class Node:
    def __init__(self, decision_attr=None, value=None, depth=None):
        self.value = value                  # Value of leaf node
        self.depth = depth                  # Depth of the current node in the tree
        self.decision_attr = decision_attr  # Attribute to split on
        self.child_nodes = {}               # Stores child nodes
        self.prob = None                    # For classification
        self.mean = None                    # For regression

    def traverse_tree(self, X, max_depth=np.inf):
        '''
        Recursive function to traverse and return value at max_depth
        '''
        # Base cases
        if self.decision_attr is None:
            return self.value
        if self.depth >= max_depth:
            return self.value

        # For classification
        if self.mean is None:
            # Check if already trained
            if X[self.decision_attr] in self.child_nodes:
                next_level = self.child_nodes[X[self.decision_attr]]
            else:
                max_prob_child, max_prob = max(self.child_nodes.items(), key=lambda x: x[1].prob)
                next_level = self.child_nodes[max_prob_child]

            return next_level.traverse_tree(X.drop(self.decision_attr), max_depth=max_depth)

        # For regression
        else:
            cur_node_mean = self.mean   
            if X[self.decision_attr] <= cur_node_mean:
                ChildNode = "low"
            else:
                ChildNode = "high"
            # Function call on the appropriate child node
            next_level = self.child_nodes[ChildNode]
            return next_level.traverse_tree(X, max_depth=max_depth)

# DecisionTree Class
@dataclass
class DecisionTree:
    def __init__(self, criterion="information_gain", max_depth=10):
        """
        Put all information to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to
        """
        self.root = None                    # Root node
        self.max_depth = max_depth          # Max depth tree can grow to, default value = 10
        self.task_type = None               # To determine classification or regression
        self.criterion = criterion          # Criteria
        self.n_samples = None               # No. of samples
        self.cols = None                    # column names

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features
        y: pd.Series with rows corresponding to the output variable
        """
        self.task_type = y.dtype
        self.n_samples = len(X)
        self.cols = y.name
        self.root = self.build_tree(X, y, None, depth=0)
        self.root.prob = 1

    def build_tree(self, X, Y, parent_node=None, depth=0):
        '''
        Recursive function to build tree.
        parent_node: caller of the function
        depth: current depth
        '''
        # Base case: all samples have the same output
        if Y.nunique() == 1:
            lc = Y.values[0]
            return Node(value=lc, depth=depth)

        # Base case: maximum depth reached or no features left
        if len(X.columns) <= 0 or depth >= self.max_depth:
            return Node(value=Y.mode(dropna=True)[0] if str(Y.dtype) == 'category' else Y.mean(), depth=depth)

        # Get the best split attribute and its mean value
        best_split_attribute, threshold = opt_split_attribute(X, Y, self.criterion, pd.Series(X.columns))

        # Create a new node
        node = Node(decision_attr=best_split_attribute)
        best_col_data = X[best_split_attribute]

        # For Discrete
        if str(best_col_data.dtype) == "category":
            X = X.drop(best_split_attribute, axis=1)  # To avoid overfitting

            # Group unique values
            best_split_classes = best_col_data.groupby(best_col_data, observed=False).count()
            for val, count in best_split_classes.items():
                frows = (best_col_data == val)  # Bool mask to filter rows
                if count > 0:
                    node.child_nodes[val] = self.build_tree(X[frows], Y[frows], node, depth + 1)
                    node.child_nodes[val].prob = len(X[frows]) / self.n_samples

        # For Real
        else:
            # Mean of the best_split_attribute
            node.mean = threshold
            # Filtering rows based on threshold
            l = (best_col_data <= threshold)
            h = (best_col_data > threshold)
            # Creating child nodes on the current node
            node.child_nodes["low"] = self.build_tree(X[l], Y[l], node, depth + 1)
            node.child_nodes["high"] = self.build_tree(X[h], Y[h], node, depth + 1)
            
            
        node.value = Y.mode(dropna=True)[0] if str(Y.dtype) == "category" else Y.mean()
        node.depth = depth
        return node

    def predict(self, X: pd.DataFrame, max_depth=np.inf) -> pd.Series:
        """
        Function to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to the output variable.
        The output variable in a row is the prediction for the sample in the corresponding row in X.
        """
        y_pred = []  # Predicted values for each row in X
        for i in X.index:
            # Prediction for each row in X
            y_pred.append(self.root.traverse_tree(X.loc[i], max_depth=max_depth))
        # Return predicted values
        return pd.Series(y_pred, name=self.cols)

    def plot(self, node=None, depth=0):
        """
        Function to plot the tree
        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        # Base case: reached leaf node
        if node is None:
            node = self.root
        if node.decision_attr is None:
            print("    " * depth + "     val = " + str(node.value) + ", depth = " + str(node.depth))
            return
        for ChildNode in node.child_nodes:
            # For classification
            if node.child_nodes[ChildNode].prob is not None:
                print("    " * depth + "  ?(X" + str(node.decision_attr) + " = " + str(ChildNode) + "):")
            # For regression
            else:
                if ChildNode == "low":
                    print("    " * depth + "  ?(X" + str(node.decision_attr) + " <= " + str(node.mean) + "):")
                elif ChildNode == "high":
                    print("    " * depth + "  ?(X" + str(node.decision_attr) + " > " + str(node.mean) + "):")
            self.plot(node.child_nodes[ChildNode], depth + 1)

