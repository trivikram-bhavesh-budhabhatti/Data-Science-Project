Machine Learning Pipeline: Diabetes and Marketing Data Analysis
Overview
This repository contains a Python script implementing a comprehensive machine learning pipeline for analyzing two datasets: "Diabetes" and "Marketing." The pipeline includes data preprocessing, unsupervised labeling, feature extraction via PCA, and a Super Learner ensemble for classification. It leverages popular libraries such as NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn.

Date: April 09, 2025

Author:

Pipeline Components
1. DataPreprocessor Class
Purpose: Cleans and prepares data for modeling.
Functions:
show_outliers: Visualizes outliers using boxplots (before/after cleaning).
clean_outliers: Removes outliers based on z-scores (threshold=3).
fill_missing: Imputes missing values with column means.
scale_data: Standardizes features using StandardScaler.
preprocess: Combines all steps, tailored for "Medical Data" (all numeric) or "Marketing Data" (select numeric columns).
Key Features: Handles missing data, scales features, and removes outliers with visualizations.
2. LabelMaker Class
Purpose: Generates labels for unsupervised data (Diabetes dataset).
Method: Uses KMeans (2 clusters) on features like Glucose, BMI, and Age; assigns "Outcome" (1 = diabetic) based on higher glucose mean.
Output: Adds binary "Outcome" column to the dataset.
3. FeatureExtractor Class
Purpose: Prepares data for modeling with train-test splits and dimensionality reduction.
Functions:
split: Splits data into training (80%) and test (20%) sets.
pca_analysis: Reduces features to 3 components using PCA.
Output: PCA-transformed training and test sets.
4. SuperLearner Class
Purpose: Builds an ensemble model for classification.
Components:
Base models: GaussianNB, MLPClassifier, KNeighborsClassifier.
Final model: DecisionTreeClassifier.
Functions:
tune_models: Hyperparameter tuning via GridSearchCV for MLP and KNN.
get_preds: Generates meta-features from base model predictions (cross-validated).
train_final: Tunes and trains the final Decision Tree on meta-features.
predict & check_accuracy: Makes predictions and computes accuracy.
Key Features: Stacking ensemble with hyperparameter optimization.
5. Encoder Class
Purpose: Encodes categorical variables (Marketing dataset).
Method: Applies LabelEncoder to columns like job, marital, etc.
Workflow
Diabetes Data
Load diabetes_project.csv.
Preprocess (clean outliers, fill missing, scale).
Generate labels using KMeans.
Split and apply PCA.
Train Super Learner and evaluate accuracy.
Marketing Data
Load Marketing.csv.
Encode categorical columns.
Preprocess numeric features.
Split, apply PCA, and evaluate with Super Learner.
Results
Outputs include cleaned data previews, PCA shapes, tuned model parameters, and accuracy scores for both datasets.
Requirements
Python 3.x
Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn
Datasets: diabetes_project.csv and Marketing.csv in ./data_file/
Installation
Clone the repository:
bash

Collapse

Wrap

Copy
git clone <repository-url>
Install dependencies:
bash

Collapse

Wrap

Copy
pip install -r requirements.txt
Ensure datasets are placed in the ./data_file/ directory.
Usage
Run the script:

bash

Collapse

Wrap

Copy
python main.py
Visualizations (e.g., outlier boxplots) require a graphical environment.
The script processes both datasets sequentially and prints results to the console.
Notes
Datasets must be in CSV format with expected column names (e.g., "Glucose" for Diabetes, "successful_marketing" for Marketing).
The code is modular and reusable for other datasets with minor adjustments.
License
This project is licensed under the MIT License - see the  file for details.

This README provides a clear, structured explanation of your project, including setup instructions and usage details, while embedding the one-pager summary you requested. Feel free to adjust the repository URL, add a requirements.txt file, or tweak the license section as needed! Let me know if you need further refinements.
