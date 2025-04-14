# Machine Learning Pipeline: Diabetes and Marketing Data Analysis

**`📌 Overview`**  
This repository contains a Python script implementing a comprehensive machine learning pipeline for analyzing two datasets: **Diabetes** and **Marketing**. The pipeline includes data preprocessing, unsupervised labeling, feature extraction via PCA, and a **Super Learner** ensemble for classification. It leverages popular libraries such as **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**, and **Scikit-learn**.

- **Author**: Nishita Vikas Shewale , Trivikram Bhavesh Budhabhatti

---

**`🔧 Pipeline Components`**

### 1. `DataPreprocessor` Class
- **Purpose**: Cleans and prepares data for modeling.
- **Functions**:
  - `show_outliers`: Visualizes outliers using boxplots (before/after cleaning).
  - `clean_outliers`: Removes outliers using z-score (threshold = 3).
  - `fill_missing`: Imputes missing values using column means.
  - `scale_data`: Standardizes features using `StandardScaler`.
  - `preprocess`: Executes all steps for either “Medical Data” or “Marketing Data”.

### 2. `LabelMaker` Class
- **Purpose**: Generates binary labels for the unsupervised Diabetes dataset.
- **Method**: Applies KMeans clustering on features like Glucose, BMI, and Age. Assigns labels based on glucose mean of clusters.

### 3. `FeatureExtractor` Class
- **Purpose**: Prepares data using PCA and performs train-test splits.
- **Functions**:
  - `split`: Splits dataset (80% train, 20% test).
  - `pca_analysis`: Applies PCA to reduce dimensions to 4 components.

### 4. `SuperLearner` Class
- **Purpose**: Ensemble model using stacking for classification.
- **Models**:
  - **Base models**: `GaussianNB`, `MLPClassifier`, `KNeighborsClassifier`
  - **Final model**: `DecisionTreeClassifier`
- **Functions**:
  - `tune_models`: Hyperparameter tuning with `GridSearchCV`.
  - `get_preds`: Generates meta-features via cross-validation.
  - `train_final`: Trains final model on stacked outputs.
  - `predict`, `check_accuracy`: Predicts and evaluates performance.

### 5. `Encoder` Class
- **Purpose**: Encodes categorical features in the Marketing dataset using `LabelEncoder`.

### 6. `DataVisualization` Class
- **Purpose**: Visualizes feature distributions, correlations, clusters, and PCA outputs.

---

## 🧐 Workflow

### 🔹 Diabetes Data
1. Load `diabetes_project.csv`
2. Visualize raw distributions and correlation matrix
3. Preprocess: clean outliers, fill missing values, scale features
4. Label using KMeans clustering
5. Visualize clusters
6. Split data and apply PCA
7. Train and evaluate Super Learner

### 🔹 Marketing Data
1. Load `Marketing.csv`
2. Encode categorical columns
3. Preprocess numeric features
4. Apply PCA, train and evaluate Super Learner

---

## 📊 Visual Outputs

This pipeline generates and saves the following plots:
- `databeforeprocessing.png`: Distribution plots of the raw diabetes dataset
- `diabetescorrelation.png`: Correlation heatmap of diabetes dataset
- `postprocessing.png`: Distribution plots after preprocessing
- Cluster visualization: Glucose vs BMI, colored by inferred Outcome
- PCA scatter plots for both datasets

---

## ✅ Project Requirement Mapping

| Step | Description | Completed |
|------|-------------|-----------|
| 1 | Preprocessing | ✅ via `DataPreprocessor` |
| 2 | Labeling (Medical) | ✅ via `LabelMaker` + KMeans |
| 3 | Feature Extraction | ✅ PCA via `FeatureExtractor` |
| 4 | Classification | ✅ `SuperLearner` stacking ensemble |
| 5 | Generalization | ✅ Applied to Marketing dataset |

---

## 🧪 Sample Output

```text
mlp best settings: {'hidden_layer_sizes': (50,), 'alpha': 0.001}
knn best settings: {'n_neighbors': 7, 'weights': 'uniform'}
Final model settings: {'max_depth': 3, 'min_samples_split': 2}
Medical Data Accuracy: 0.8759
Marketing Data Accuracy: 0.8884
```

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
## 📁 Code Structure

- `main.py` — Entry point executing full pipeline
- `DataPreprocessor` — Data cleaning and scaling
- `LabelMaker` — Generates labels for diabetes data
- `FeatureExtractor` — PCA and splitting
- `SuperLearner` — Ensemble classification
- `Encoder` — Encodes marketing data
- `DataVisualization` — Generates all figures

---

**`📝 Notes`**
- Datasets must follow expected formats with columns like `Glucose`, `BMI`, `successful_marketing`, etc.
- Modular design allows easy adaptation to other datasets with minimal changes.

---

**`📄 License`**  
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
