# Heart Disease Prediction

## Project Overview
This project aims to predict the presence of heart disease (Cardiovascular Diseases - CVDs) based on various health indicators. Using machine learning models, specifically Decision Trees and Random Forests, we analyze a dataset to identify patterns and build a predictive tool. The goal is to assist in early detection and risk assessment of heart disease.

## Dataset Information
This dataset is obtained from Kaggle: [Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

## Setup and Dependencies
To run this project, you will need the following Python libraries. They can typically be installed using `pip`.

*   `numpy`
*   `pandas`
*   `sklearn.tree.DecisionTreeClassifier`
*   `sklearn.ensemble.RandomForestClassifier`
*   `sklearn.model_selection.train_test_split`
*   `sklearn.metrics.accuracy_score`
*   `xgboost.XGBClassifier` (though not explicitly used for final models, it was imported)
*   `matplotlib.pyplot`

## Data Preprocessing
The data preprocessing steps involved:
1.  **Loading the Dataset**: The `heart.csv` file was loaded into a pandas DataFrame using `pd.read_csv("/content/heart.csv")`.
2.  **Identifying Categorical Features**: The following columns were identified as categorical and suitable for one-hot encoding:
    *   `Sex`
    *   `ChestPainType`
    *   `RestingECG`
    *   `ExerciseAngina`
    *   `ST_Slope`
3.  **One-Hot Encoding**: These categorical features were converted into numerical format using pandas' `pd.get_dummies` function. The original DataFrame `df` was updated with the one-hot encoded columns, using a prefix for clarity (e.g., `Sex_M`, `Sex_F`).
4.  **Feature Selection**: A list of features (`features`) was created by excluding the target variable `HeartDisease` from the DataFrame columns.
5.  **Data Splitting**: The dataset was split into training and validation sets to evaluate model performance. `train_test_split` from `sklearn.model_selection` was used with an 80% training size (`train_size=0.8`) and `random_state=55` for reproducibility. The target variable was `HeartDisease`.

## Model Building and Evaluation

### 1. Decision Tree Classifier
We built a Decision Tree Classifier and tuned its hyperparameters to find an optimal model. The following hyperparameters were explored:
*   `min_samples_split`: `[2, 10, 30, 50, 100, 200, 300, 700]`
*   `max_depth`: `[1, 2, 3, 4, 8, 16, 32, 64, None]`

After evaluating various combinations, the chosen parameters for the modest Decision Tree model were:
*   `min_samples_split = 50`
*   `max_depth = 4`

The evaluation metrics for this model were:
*   **Training Accuracy**: 0.8665
*   **Validation Accuracy**: 0.8696

### 2. Random Forest Classifier
We also developed a Random Forest Classifier, exploring a wider range of hyperparameters to enhance predictive power and reduce overfitting. The hyperparameters tuned included:
*   `min_samples_split`: `[2, 10, 50, 100, 200, 300, 700]`
*   `max_depth`: `[2, 4, 9, 16, 32, 64, None]`
*   `n_estimators`: `[10, 50, 100, 500]`

Based on the tuning process, the final chosen parameters for the Random Forest model were:
*   `n_estimators = 100`
*   `max_depth = 16`
*   `min_samples_split = 10`

The evaluation metrics for this model were:
*   **Training Accuracy**: 0.9292
*   **Validation Accuracy**: 0.8967

## Results and Conclusion
Comparing the validation accuracies of both models:
*   **Decision Tree Classifier**: 0.8696
*   **Random Forest Classifier**: 0.8967

The **Random Forest Classifier** demonstrated superior performance with a validation accuracy of 0.8967, compared to the Decision Tree's 0.8696. This indicates that the ensemble nature of Random Forest, by combining multiple decision trees, provides a more robust and accurate prediction for heart disease in this dataset.

## How to Run the Code
1.  **Environment**: The code is designed to be run in a Google Colab environment. 
2.  **Dataset**: Ensure that the `heart.csv` dataset is uploaded to your Colab environment or is accessible at the path `/content/heart.csv`. 
3.  **Execution**: Simply run the cells sequentially in the provided Jupyter Notebook. The notebook handles all necessary imports, data loading, preprocessing, model training, and evaluation steps.
