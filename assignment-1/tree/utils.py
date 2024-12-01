"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import math
import numpy as np

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if (str(y.dtype) == 'float64' or str(y.dtype) == 'int64'):
        return True
    else:
        return False
    pass

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    # Probability of each discrete value
    probability = Y.value_counts(normalize=True)
    entropy_value = 0
    for p in probability:
        if p>0:
            entropy_value+= -1 * p * math.log2(p + 1e-10)

    return entropy_value
    pass

def mse(Y:pd.Series) -> float:
    mean_value = Y.mean()
    mse_value = ((Y - mean_value) ** 2).mean()
    return mse_value
    pass


def information_gain(Y: pd.Series, attr: pd.Series) -> float:
    """
    Function to calculate the information gain
    """
    assert Y.size == attr.size

    # Data is DISCRETE INPUT REAL OUTPUT (DIRO)
    if check_ifreal(attr)==False and check_ifreal(Y)==True:
        # Combine Y and attr into a DataFrame
        df = pd.DataFrame({'Y': Y, 'Attr': attr})
        # Calculate the overall MSE
        overall_mse = mse(Y)
        # Calculate the weighted MSE of the subsets
        weighted_mses = df.groupby('Attr', observed=False)['Y'].apply(mse) * df.groupby('Attr', observed=False).size() / len(df)
        # Calculate the MSE reduction
        mse_reduction_value = overall_mse - weighted_mses.sum()
        return mse_reduction_value
    
    # Data is DISCRETE INPUT DISCRETE OUTPUT (DIDO)
    elif check_ifreal(attr)==False and check_ifreal(Y)==False:
        # Combine Y and attr into a DataFrame
        df = pd.DataFrame({'Y': Y, 'Attr': attr})
        # Calculate the overall entropy
        overall_entropy = entropy(Y)
        # Calculate the weighted entropy of the subsets
        weighted_entropies = df.groupby('Attr', observed=False)['Y'].apply(entropy) * df.groupby('Attr', observed=False).size() / len(df)
        # Calculate the information gain
        information_gain_value = overall_entropy - weighted_entropies.sum()
        return information_gain_value
    
    # Data is REAL INPUT DISCRETE OUTPUT (RIDO)
    elif check_ifreal(attr)==True and check_ifreal(Y)==False:
        df = pd.DataFrame({'Y': Y, 'attr':attr})
        best_split_val = None
        max_gain = -np.inf
        attr_sorted = attr.sort_values()
        df_sorted = df.sort_values(by = 'attr')
        low = attr_sorted.index[0]
        # finding best split and best gain by spliting with mid values of each consequtive values in attr_sorted
        for high in attr_sorted.index[1:]:
            mid = (attr_sorted[low] + attr_sorted[high])/2

            overall_entropy = entropy(Y)
            weighted_entropy_left = df_sorted.loc[attr <= mid]['Y'].count() * entropy(df_sorted.loc[attr <= mid]['Y']) / len(attr)
            weighted_entropy_right =  df_sorted.loc[attr > mid]['Y'].count() * entropy(df_sorted.loc[attr > mid]['Y']) / len(attr)
            gain = overall_entropy - (weighted_entropy_left+weighted_entropy_right)
      
            # Storing the best_gain and best_split_val
            if gain > max_gain:
                max_gain = gain
                best_split_val = mid
            
            low = high
        information_gain_value = (max_gain, best_split_val)
        return information_gain_value

    # Data is REAL INPUT Real OUTPUT (RIRO)
    else:
        df = pd.DataFrame({'Y': Y,'attr':attr})
        #print(df)
        best_split_val = None
        max_gain = -np.inf
        attr_sorted = attr.sort_values()
        df_sorted = df.sort_values(by = 'attr')
        low = attr_sorted.index[0]
        # finding best split and best gain by spliting with mid values of each consequtive values in attr_sorted
        for high in attr_sorted.index[1:]:
            mid = (attr_sorted[low] + attr_sorted[high])/2

            overall_mse = entropy(Y)
            weighted_mse_left = df_sorted.loc[attr <= mid]['Y'].count() * mse(df_sorted.loc[attr <= mid]['Y']) / len(attr)
            weighted_mse_right =  df_sorted.loc[attr > mid]['Y'].count() * mse(df_sorted.loc[attr > mid]['Y']) / len(attr)
            gain = overall_mse - (weighted_mse_left+weighted_mse_right)
      
            # Storing the best_gain and best_split_val
            if gain > max_gain:
                max_gain = gain
                best_split_val = mid
            
            low = high
        information_gain_value = (max_gain, best_split_val)
        return information_gain_value
    pass

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    # probability of each class
    probability = Y.value_counts(normalize=True)
    # Calculate Gini index
    s=0
    for p in probability:
        s+= p**2
    gini = 1 - s
    return gini
    pass



def gini_gain(Y, attr):
    df = pd.DataFrame({'Y': Y,'attr':attr})
    # For DISCRETE INPUT DISCRETE OUTPUT (DIDO)
    Weighted_gini_gain = 0
    if check_ifreal(attr) == False and check_ifreal(Y) == False:       
        unique_values = list(attr.unique())
        overall_gini_gain = gini_index(Y)
        df = pd.DataFrame({'Y': Y, 'attr':attr})
        for v in unique_values:
            Weighted_gini_gain += df[attr == v]['Y'].count() * gini_index(df.loc[attr == v]['Y']) / len(attr)
        Total_gini_gain = overall_gini_gain - Weighted_gini_gain 
        return Total_gini_gain

    # For REAL INPUT DISCRETE OUTPUT (RIDO)
    elif check_ifreal(attr)==True and check_ifreal(Y)==False:
        best_split_val = None
        max_gain = -np.inf
        attr_sorted = attr.sort_values()    
        df_sorted = df.sort_values(by = 'attr')
        low = attr_sorted.index[0]  

        for high in attr_sorted.index[1:]:
            mid = (attr_sorted[low] + attr_sorted[high])/2  #


            overall_gini = gini_index(Y)
            weighted_gini_left = df_sorted.loc[attr <= mid]['Y'].count() * gini_index(df_sorted.loc[attr <= mid]['Y']) / len(attr)
            weighted_gini_right =  df_sorted.loc[attr > mid]['Y'].count() * gini_index(df_sorted.loc[attr > mid]['Y']) / len(attr)
            gain = overall_gini - (weighted_gini_left+weighted_gini_right)

            # Storing the best_gain and best_split_val
            if gain > max_gain:
                max_gain = gain
                best_split_val = mid
            low = high
        Total_gini_gain = (max_gain, best_split_val)
        return Total_gini_gain



def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon and threshold value
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    """

    Parameters:
    - X: DataFrame with real-valued features.
    - y: Series with discrete output (target variable).
    - criterion: Splitting criterion ('gini' or 'entropy').
    """

    if criterion not in ["gini_index", "information_gain"]:
        raise ValueError("Invalid criterion. Supported values are 'gini_index' and 'information_gain'.")
    

    # determining the best column
    max_ig = -np.inf  # highest information gain
    best_threshold = None  # mean of the feature with the highest IG
    best_split_attribute = None  # feature with the highest IG
    for c in list(X.columns):
        col_split_val = None
        cur_col_ig = None

        if check_ifreal(y)==False and criterion == "gini_index":
            column_ig = gini_gain(y, X[c])
        else:
            column_ig = information_gain(y, X[c])

        cur_col_ig = column_ig

        # real/continuous values [information gain, mean]
        if type(column_ig) == tuple:
             cur_col_ig, col_split_val = column_ig[0], column_ig[1]

        if cur_col_ig > max_ig:
            max_ig = cur_col_ig
            best_threshold = col_split_val
            best_split_attribute = c  # feature with the highest IG

    
    return best_split_attribute, best_threshold
    pass


# # def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
# #     """
# #     Funtion to split the data according to an attribute.
# #     If needed you can split this function into 2, one for discrete and one for real valued features.
# #     You can also change the parameters of this function according to your implementation.

# #     attribute: attribute/feature to split upon
# #     value: value of that attribute to split upon

# #     return: splitted data(Input and output)
# #     """

# #     # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.


