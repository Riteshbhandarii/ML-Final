# Bank Marketing Campaign: Neural Network Model Documentation

## Project Overview

This project aims to predict whether a client will subscribe to a term deposit based on various features including demographic data, economic indicators, and previous marketing campaign interactions. The dataset contains 41,188 rows and 21 columns with a mix of categorical and numerical features.

## Table of Contents

1. [Data Processing](#data-processing)
2. [Feature Engineering](#feature-engineering)
3. [Feature Selection](#feature-selection)
4. [Model Development](#model-development)
5. [Model Evaluation](#model-evaluation)
6. [Rationale for Model Selection](#rationale-for-model-selection)
7. [Conclusion](#conclusion)

## Data Processing

### Initial Data Cleaning
- Removed 12 duplicate records
- Handled missing values (labeled as 'unknown') in categorical columns by replacing them with the mode
- The dataset had a significant number of 'unknown' values, particularly in the 'default' column (8,597)

### Data Transformation
- Binary target encoding: 'yes'/'no' to 1/0
- Numerical features: 
  - StandardScaler for: age, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed
  - MinMaxScaler for: duration, campaign, pdays, previous
- Categorical features:
  - One-hot encoding for: job, marital, default, housing, loan, contact, day_of_week, poutcome
  - Cyclical encoding for month using sine and cosine transformations
  - Ordinal encoding for education levels

## Feature Engineering

The project employed several techniques to transform the raw data:

1. **Cyclic features**: Month data was transformed using sine and cosine functions to preserve the cyclic nature of months
2. **Education level mapping**: Education levels were mapped on an ordinal scale from 0 (illiterate) to 6 (university degree)

## Feature Selection

Features were selected based on:

1. **Correlation analysis**: Highly correlated features were identified and removed to reduce multicollinearity
2. **Feature importance**: Random Forest was used to identify the top features that explain 90% of the variance

The final feature set included 19 features:
- age
- nr.employed
- campaign
- education
- housing_yes
- marital_married
- poutcome_success
- loan_yes
- cons.conf.idx
- day_of_week_thu
- day_of_week_tue
- month_sin
- day_of_week_mon
- day_of_week_wed
- job_technician
- default_no
- cons.price.idx
- job_blue-collar
- month_cos

## Model Development

### Neural Network Architecture
- Input layer: 19 neurons (for each feature)
- Hidden layer 1: 128 neurons with ReLU activation and BatchNormalization, Dropout(0.3)
- Hidden layer 2: 64 neurons with ReLU activation and BatchNormalization, Dropout(0.2)
- Hidden layer 3: 32 neurons with ReLU activation and BatchNormalization, Dropout(0.2)
- Output layer: 1 neuron with sigmoid activation for binary classification

### Hyperparameter Tuning
- Batch size: [32, 64, 128]
- Learning rate: [0.001, 0.01]
- Dropout rate: [0.2, 0.3]
- Optimizer: [Adam, RMSprop]

Best parameters:
- Batch size: 32
- Learning rate: 0.001
- Dropout rate: 0.2
- Optimizer: Adam

### Training Strategy
- Early stopping with patience=5
- ReduceLROnPlateau to dynamically adjust learning rate
- Training for up to 150 epochs but typically stopping earlier due to early stopping

## Model Evaluation

### Performance Metrics
- **Accuracy**: 0.8932
- **ROC AUC Score**: 0.7609
- **Precision**: 
  - Class 0 (No subscription): 0.90
  - Class 1 (Subscription): 0.69
- **Recall**:
  - Class 0: 0.99
  - Class 1: 0.16
- **F1-score**:
  - Class 0: 0.94
  - Class 1: 0.26

### Cross-Validation Results
- 5-fold CV Accuracy: 0.8989 (±0.0027)

### Confusion Matrix
```
[[3601   35]
 [ 405   77]]
```

## Rationale for Model Selection

The Artificial Neural Network (ANN) model was selected as our final model due to several key considerations:

**1. Complex Feature Relationships**
- The dataset contained a mix of categorical and numerical features with potentially non-linear relationships
- ANNs excel at learning complex patterns that might not be captured by simpler models like logistic regression

**2. Performance Metrics**
- The model achieved an overall accuracy of 89.32%, which is substantial for this imbalanced dataset
- AUC-ROC score of 0.7609 indicates good discriminative ability between subscribers and non-subscribers
- Cross-validation results showed consistent performance across different data splits with low standard deviation (±0.0027)

**3. Class Imbalance Handling**
- Despite the significant class imbalance in the dataset, the model maintained reasonable performance
- While recall for the minority class (subscribers) is low at 16%, this is a common challenge with imbalanced datasets
- The model successfully identified 77 potential subscribers out of 482, which provides valuable leads for marketing efforts

**4. Model Stability**
- Regularization techniques like Dropout and BatchNormalization helped prevent overfitting
- Learning rate scheduling through ReduceLROnPlateau allowed for stable convergence

**5. Practical Applications**
- In a marketing context, even modest improvements in identifying potential subscribers can lead to significant business impact
- The model provides probability scores that can be used to prioritize marketing efforts

**6. Limitations**
- The low recall for the positive class (16%) means many potential subscribers would be missed
- This suggests that for a real-world application, the threshold might need adjustment to increase sensitivity at the expense of some precision

The ANN model provides a balance between accuracy and generalizability that aligns with the project's goal of identifying potential term deposit subscribers. While there is room for improvement in minority class detection, the model serves as a solid foundation for the bank's marketing strategy.

## Conclusion

The developed neural network model provides a reliable prediction system for the bank's marketing team to identify potential term deposit subscribers. The model's strength lies in its ability to process multiple feature types and learn complex relationships in the data.

To further improve performance, especially for recall of the positive class (subscribers), future work could explore:

1. Advanced techniques for handling class imbalance (SMOTE, class weights, etc.)
2. Ensemble methods combining the ANN with other models
3. Feature engineering to develop more predictive variables
4. Adjusting the classification threshold to favor recall over precision

Despite these potential improvements, the current model offers valuable insights for marketing strategy optimization and resource allocation.
