1. Model Performance Comparison:
    - Decision Tree Accuracy: 0.81
    - Random Forest Accuracy: 0.92
    - Linear Regression Accuracy: 0.03
      
2. Effectiveness of Decision Tree and Random Forest:
   - Decision trees and random forests are classification algorithms inherently designed for handling categorical labels.
   - They construct a series of rules based on features to partition the data into classes. Random forests further enhance the performance (higher accuracy than DT) by creating multiple decision trees and aggregating their outputs.
     
3. Is the usage of linear regression for classification justified? and why? 
    - Answer: No.
    - Linear Regression is designed for predicting continuous numerical values, not categorical labels.
    - It assumes a linear relationship between features and the target variable, which may not hold true for classification tasks with categorical labels.
    - Linear Regression may not capture the underlying patterns and relationships effectively for categorical label prediction, as it assumes a linear relationship between features and the target variable, which may not hold true for classification tasks with categorical labels.
    - Even after using N-1 encoding accuracy (0.22) is not as significant as in case for DT and RF.
