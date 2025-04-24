# Bank Marketing Prediction with Artificial Neural Networks

## Project Overview
This project aims to predict whether a client will subscribe to a term deposit based on data from a bank’s marketing campaigns. Using an Artificial Neural Network (ANN), we leverage the Bank Marketing dataset from the UCI Machine Learning Repository to build a robust classification model.

## Dataset
- **Source**: [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- **Size**: 41,188 instances
- **Features**: 20 input features (10 numerical, 10 categorical)
- **Target**: Binary (`yes`/`no`) indicating subscription to a term deposit

## Methodology
1. **Data Loading**:
   - Loaded with error handling to ensure robustness.
   
2. **Cleaning & Preprocessing**:
   - Removed duplicates after inspection.
   - Imputed 'unknown' values with mode for most features; used domain knowledge for 'default'.

3. **Exploratory Data Analysis (EDA)**:
   - Visualized numerical features (e.g., histograms).
   - Analyzed categorical features with bar plots (e.g., `job`, `education`).

4. **Feature Engineering**:
   - Applied standard scaling to numerical features and one-hot encoding to categorical features using a scikit-learn pipeline.

5. **Feature Selection**:
   - Used Recursive Feature Elimination (RFE) with a Random Forest Classifier to select the top 15 features.

6. **Model Building**:
   - Designed an ANN with Keras Tuner for hyperparameter optimization (e.g., layer sizes, dropout rates, learning rates).
   - Addressed class imbalance with class weights.

7. **Training & Evaluation**:
   - Trained with 50 epochs and early stopping.
   - Evaluated using AUC-ROC, precision, recall, and F1-score alongside accuracy.

## Results
- **Final Test AUC-ROC**: ~0.76 (varies slightly with tuning)
- **Accuracy**: ~0.89
- **Classification Report** (example from original run):
  ```
                precision    recall  f1-score   support
           0       0.90      0.99      0.94      3636
           1       0.69      0.16      0.26       482
    accuracy                           0.89      4118
   macro avg       0.79      0.58      0.60      4118
weighted avg       0.87      0.89      0.86      4118
  ```
- **Key Insight**: The model excels at predicting non-subscribers but struggles with subscribers due to class imbalance.

## How to Run
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**:
   - Open `ML-Final-Improved.ipynb` in Jupyter Notebook or JupyterLab.
   - Ensure `bank-additional-full.csv` is in the same directory.
   - Execute all cells sequentially.

## Dependencies
- Python 3.11
- pandas
- numpy
- scikit-learn
- tensorflow
- keras-tuner
- matplotlib
- seaborn

## Files
- `ML-Final-Improved.ipynb`: Main notebook with improved code.
- `bank-additional-full.csv`: Dataset (download from UCI link).
- `numerical_hist.png`, `job_distribution.png`, `education_distribution.png`, `roc_curve.png`, `training_history.png`: Generated plots.

## Future Improvements
- Experiment with SMOTE or oversampling to further address class imbalance.
- Explore additional feature engineering (e.g., interaction terms).
- Test alternative models (e.g., XGBoost, LightGBM) for comparison.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Dataset provided by UCI Machine Learning Repository.
- Built as part of a Machine Learning course final project.
