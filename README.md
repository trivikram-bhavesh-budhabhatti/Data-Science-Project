# Machine Learning Pipeline: Diabetes and Marketing Data Analysis

**`📌 Overview`**  
This repository contains a Python script implementing a comprehensive machine learning pipeline for analyzing two datasets: **Diabetes** and **Marketing**. The pipeline includes data preprocessing, unsupervised labeling, feature extraction via PCA, and a **Super Learner** ensemble for classification. It leverages popular libraries such as **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**, and **Scikit-learn**.

- **Author**: Nishita Vikas Shewale , Trivikram Bhavesh Budhabhatti

---

**`🔧 Pipeline Components`**

**`1. DataPreprocessor Class`**
- **Purpose**: Cleans and prepares data for modeling.
- **Functions**:
  - `show_outliers`: Visualizes outliers using boxplots (before/after cleaning).
  - `clean_outliers`: Removes outliers using z-score (threshold = 3).
  - `fill_missing`: Imputes missing values using column means.
  - `scale_data`: Standardizes features using `StandardScaler`.
  - `preprocess`: Executes all steps, tailored to either “Medical Data” or “Marketing Data”.
- **Key Features**: Handles missing values, scales data, removes outliers with optional visualizations.

**`2. LabelMaker Class`**
- **Purpose**: Generates binary labels for the unsupervised Diabetes dataset.
- **Method**: 
  - Applies `KMeans` (2 clusters) on features like Glucose, BMI, and Age.
  - Assigns “Outcome” based on higher glucose mean in clusters.

**`3. FeatureExtractor Class`**
- **Purpose**: Prepares data using PCA and train-test splits.
- **Functions**:
  - `split`: Splits dataset (80% train, 20% test).
  - `pca_analysis`: Applies PCA to reduce dimensions to 3 components.

**`4. SuperLearner Class`**
- **Purpose**: Ensemble model using stacking for classification.
- **Models**:
  - **Base models**: `GaussianNB`, `MLPClassifier`, `KNeighborsClassifier`
  - **Final model**: `DecisionTreeClassifier`
- **Functions**:
  - `tune_models`: Hyperparameter tuning with `GridSearchCV`.
  - `get_preds`: Generates meta-features via cross-validation.
  - `train_final`: Trains the final model on stacked outputs.
  - `predict`, `check_accuracy`: Makes predictions and evaluates performance.

**`5. Encoder Class`**
- **Purpose**: Encodes categorical features in the Marketing dataset using `LabelEncoder`.

---

**`🧠 Workflow`**

**🔹 Diabetes Data**
1. Load `diabetes_project.csv`.
2. Preprocess: clean outliers, fill missing, scale features.
3. Label: Use KMeans to infer diabetic status.
4. Split and reduce dimensions using PCA.
5. Train and evaluate the Super Learner.

**🔹 Marketing Data**
1. Load `Marketing.csv`.
2. Encode categorical columns.
3. Preprocess numeric features.
4. Apply PCA, train and evaluate using Super Learner.

---

**`📊 Results`**
- Cleaned dataset previews  
- PCA-transformed shapes  
- Best model hyperparameters  
- Classification accuracy for both datasets  

---

**`📦 Requirements`**
- Python 3.x  
- Libraries:  
  ```
  numpy  
  pandas  
  matplotlib  
  seaborn  
  scikit-learn
  ```
- Datasets:  
  Place `diabetes_project.csv` and `Marketing.csv` inside the `./data_file/` directory.

---

**`⚙️ Installation`**

```bash
git clone <repository-url>
cd <repo-folder>
pip install -r requirements.txt
```

---

**`🚀 Usage`**

```bash
python main.py
```
---

**`📝 Notes`**
- Datasets must follow expected formats with columns like `Glucose`, `BMI`, `successful_marketing`, etc.
- Modular design allows easy adaptation to other datasets with minimal changes.

---

**`📄 License`**  
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
