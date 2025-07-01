
# PRODIGY_ML_Task1
**Task-01:-**  Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.

**Dataset:-** https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

## **Libraries used:-**
* pandas: Data manipulation
* numpy: Numerical operations
* matplotlib.pyplot: Plotting on graph
* sklearn: Linear modelling of dataset
* scipy: Skew calculations

## **Column used for Training and Testing:-**
* LotArea
* BedroomAbvGr
* BsmtFullBath
* BsmtHalfBath
* HalfBath
* TotalBsmtSF
* FullBath

## **Data Split:-**
* The target variable, 'SalePrice', undergoes a logarithmic transformation using np.log1p(). This is done to handle potential skewness in the target variable and to make the predictions more interpretable.
* The training set is split into features (X) and the target variable (y).
* Further, the data is split into training and testing subsets using train_test_split(). This function randomly splits the data into training and testing sets, typically in a 2:1 ratio in this case, with 33% of the data reserved for testing. The random state is set to ensure reproducibility.

## **Feature Scaling:-**
* Standard scaling is applied to the feature variables using StandardScaler(). This ensures that all features have a mean of 0 and a standard deviation of 1, which helps in improving the performance of many machine learning algorithms.

## **Missing Value Imputation:-**
* Missing values in the test set are imputed using the mean strategy (strategy='mean') through SimpleImputer(). The mean values are calculated from the training data and used to fill in missing values in both the training and testing sets.

## **Model Training:-**
* A Ridge regression model is trained using the training data. Ridge regression is a linear regression model with a regularization term, which helps in reducing overfitting. The regularization strength (alpha) is set to 1.0, but it can be tuned for optimal performance.

## **Model Evaluation:-**
* The trained model is used to make predictions on the testing set.
* Mean Squared Error (MSE) is calculated to evaluate the performance of the model. MSE measures the average squared difference between the predicted values and the actual values.

## **Visualization:**
* The code creates scatter plots to visualize the relationship between actual sale prices and predicted sale prices, as well as between lot area and predicted sale prices for the test data.
