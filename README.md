# Bank Marketing ANN Model

## Project Overview
This project develops a neural network model to predict whether a bank marketing campaign will successfully convince a client to subscribe to a term deposit. The model analyzes various client attributes, communication details, and economic indicators to make these predictions.

## Dataset Description
The dataset contains information from a Portuguese banking institution's direct marketing campaigns. It includes:

- **Client demographics**: age, job, marital status, education, etc.
- **Contact information**: communication type, month and day of contact
- **Campaign details**: number of contacts, previous campaign outcomes
- **Economic indicators**: employment variation rate, consumer price index, etc.
- **Target variable**: whether the client subscribed to a term deposit (yes/no)

## Project Structure
1. **Data Loading & Exploration**: Examination of data characteristics, missing values and distributions
2. **Data Preprocessing**: 
   - Handling unknown values
   - Feature scaling with StandardScaler and MinMaxScaler
   - Categorical encoding (one-hot and cyclical encoding)
3. **Feature Engineering**:
   - Cyclical encoding for month and education
   - Feature selection based on correlation analysis
   - Feature importance ranking using Random Forest
4. **Model Development**:
   - Neural network architecture with 3 hidden layers
   - Regularization with dropout
   - Binary classification with sigmoid activation
5. **Model Training & Evaluation**:
   - Cross-validation
   - Hyperparameter tuning
   - Performance metrics (accuracy, precision, recall, F1-score, ROC-AUC)

## Key Findings
- Achieved 89.3% accuracy on test data
- Important features include age, employment indicators, and campaign contact details
- Model shows high precision (90%) but low recall (16%) for positive class
- Economic indicators demonstrate strong correlation with term deposit subscription likelihood

## Model Performance
- **Accuracy**: 89.3%
- **AUC**: 0.76
- **Precision** (class 1): 69%
- **Recall** (class 1): 16%
- **F1-score** (class 1): 26%

## Future Improvements
- Address class imbalance through resampling or class weights
- Explore more complex feature engineering
- Try ensemble methods or other model architectures
- Optimize for business metrics (e.g., campaign ROI) rather than just accuracy

## Requirements
- Python 3.x
- TensorFlow
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Usage
1. Load the dataset using pandas
2. Run preprocessing steps to clean and transform data
3. Train the model with the provided hyperparameters
4. Evaluate model performance on test data
5. Use the model to predict subscription likelihood for new marketing contacts

## Business Application
This model helps banking institutions optimize their marketing campaigns by:
- Targeting clients with higher likelihood of subscription
- Reducing unnecessary contacts to improve efficiency
- Understanding key factors that influence subscription decisions
- Adapting campaign strategies based on feature importance
