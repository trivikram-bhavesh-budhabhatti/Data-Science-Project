import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Data Preprocessing Class
class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def show_outliers(self, df, cols, when="Before", name="Data"):
        print(f"Checking outliers for {name} - {when}")
        plt.figure(figsize=(20, 5))
        for i, col in enumerate(cols, 1):
            plt.subplot(1, len(cols), i)
            sns.boxplot(y=df[col], color='blue')
            plt.title(col)
            plt.xticks([])
        plt.suptitle(f"Outliers {when} Cleaning ({name})")
        plt.tight_layout()
        plt.show()

    def clean_outliers(self, df, cols, thresh=3, name="Data"):
        self.show_outliers(df, cols, "Before", name)
        for col in cols:
            z = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z < thresh]
        self.show_outliers(df, cols, "After", name)
        return df

    def fill_missing(self, df, cols):
        df[cols] = self.imputer.fit_transform(df[cols])
        return df

    def scale_data(self, df, cols):
        df[cols] = self.scaler.fit_transform(df[cols])
        return df

    def preprocess(self, df, name="Data"):
        print(f"Starting preprocessing for {name}")
        cols = df.columns
        df = self.fill_missing(df, cols)
        if name == "Diabetes Data":
            num_cols = cols
        else:
            num_cols = ['age', 'last_contact_duration', 'campaign_contacts',
                        'passed_days', 'previous_contacts']
            num_cols = [c for c in num_cols if c in df.columns]
        df = self.clean_outliers(df, num_cols, name=name)
        df = self.scale_data(df, cols)
        return df


# Data Visualization Class
class DataVisualization:
    def visualize_diabetes_data(self, df):
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(df.columns, 1):
            plt.subplot(3, 3, i)
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.suptitle("Diabetes Dataset Feature Distributions", y=1.02)
        plt.savefig("databeforeprocessing.png")

        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title("Correlation Matrix of Diabetes Dataset")
        plt.savefig('diabetescorrelation.png')

    def visualize_diabetes_data_post_preprocessing(self, df):
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(df.columns, 1):
            plt.subplot(3, 3, i)
            sns.histplot(df[col], kde=True)
            plt.title(f'Normalized {col}')
        plt.tight_layout()
        plt.suptitle("Diabetes Dataset After Preprocessing", y=1.02)
        plt.savefig('postprocessing.png')

    def visualize_marketing_data_preprocessing(self, df):
        numeric_cols = df.select_dtypes(include=np.number).columns
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(df[col], kde=True)
            plt.title(f'Marketing - Raw {col}')
        plt.tight_layout()
        plt.suptitle("Marketing Data Feature Distributions (Before Preprocessing)", y=1.02)
        plt.savefig("marketing_before_processing.png")
        # plt.show()

        exclude_cols = ['balance', 'default', 'marital', 'loan', 'previous_contacts']

        # Filter numeric columns and drop unwanted ones
        numeric_cols1 = df.select_dtypes(include=np.number).columns
        numeric_cols1 = [col for col in numeric_cols1 if col not in exclude_cols]
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numeric_cols1].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Marketing Data Correlation Matrix (Before Preprocessing)", fontsize=14, pad=12)
        plt.savefig("marketing_corr_before.png")
        # plt.show()

    def visualize_marketing_data_post_preprocessing(self, df):
        numeric_cols = df.select_dtypes(include=np.number).columns
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols  # auto-calculate rows

        plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(df[col], kde=True, color='lightgreen')
            plt.title(f'Marketing - Normalized {col}')
            plt.xlabel('')
            plt.ylabel('')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, left=0.05, right=0.95, bottom=0.05)
        plt.suptitle("Marketing Data Feature Distributions (After Preprocessing)", y=1.02)
        plt.savefig("marketing_after_processing.png")

    def visualize_pca(self, X_pca, y, title="PCA Visualization"):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='Set1')
        plt.title(title)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def visualize_clusters(self, df, features, label_col='Outcome'):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=features[0], y=features[1], hue=label_col, palette='Set1')
        plt.title(f"Label Generation: {features[0]} vs {features[1]}")
        plt.grid(True)
        plt.show()

    def visualize_cumulative_pca_variance(self, pca, title="PCA Explained Variance"):
        plt.figure(figsize=(8, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', color='blue')
        plt.title(f'Cumulative Explained Variance - {title}')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.tight_layout()
        plt.show()



# Label Maker Class
class LabelMaker:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=2, random_state=42)

    def make_labels(self, df, feats):
        clusters = self.kmeans.fit_predict(df[feats])
        df['Cluster'] = clusters
        means = df.groupby('Cluster')['Glucose'].mean()
        diabetic = means.idxmax()
        df['Outcome'] = df['Cluster'].apply(lambda x: 1 if x == diabetic else 0)
        df = df.drop('Cluster', axis=1)
        print("Data with labels:")
        print(df.head())
        return df


# Feature Extraction Class
class FeatureExtractor:
    def __init__(self):
        self.pca = PCA(n_components=4)

    def split(self, df, target):
        X = df.drop(target, axis=1)
        y = df[target]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def pca_analysis(self, X_train, X_test):
        return self.pca.fit_transform(X_train), self.pca.transform(X_test)


# Super Learner Class
class SuperLearner:
    def __init__(self):
        self.models = [
            ('nb', GaussianNB()),
            ('mlp', MLPClassifier(max_iter=2000, random_state=42)),
            ('knn', KNeighborsClassifier())
        ]
        self.final_model = DecisionTreeClassifier(random_state=42)

    def tune_models(self, X, y):
        params = {
            'mlp': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001]},
            'knn': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
        }
        tuned = []
        for name, model in self.models:
            if name in params:
                grid = GridSearchCV(model, params[name], cv=5, scoring='accuracy')
                grid.fit(X, y)
                print(f"{name} best settings: {grid.best_params_}")
                tuned.append((name, grid.best_estimator_))
            else:
                model.fit(X, y)
                tuned.append((name, model))
        self.models = tuned

    def get_preds(self, X, y):
        return np.column_stack([cross_val_predict(model, X, y, cv=5) for _, model in self.models])

    def train_final(self, X_meta, y):
        grid = GridSearchCV(self.final_model, {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]},
                            cv=5, scoring='accuracy')
        grid.fit(X_meta, y)
        self.final_model = grid.best_estimator_
        print(f"Final model settings: {grid.best_params_}")

    def predict(self, X):
        return self.final_model.predict(np.column_stack([m.predict(X) for _, m in self.models]))

    def check_accuracy(self, X, y):
        return accuracy_score(y, self.predict(X))


# Encoder Class
class Encoder:
    def __init__(self):
        self.encoders = {}

    def encode(self, df, cols):
        for col in cols:
            self.encoders[col] = LabelEncoder()
            df[col] = self.encoders[col].fit_transform(df[col].astype(str))
        return df


# Main Function
def main():
    df = pd.read_csv("./data_file/diabetes_project.csv")
    print(df.describe())
    print("Diabetes columns:", df.columns.tolist())
    viz = DataVisualization()
    viz.visualize_diabetes_data(df)

    prep = DataPreprocessor()
    df_clean = prep.preprocess(df, "Diabetes Data")
    viz.visualize_diabetes_data_post_preprocessing(df_clean)

    label_maker = LabelMaker()
    feats = ['Glucose', 'BMI', 'Age']
    df_with_labels = label_maker.make_labels(df_clean, feats)
    viz.visualize_clusters(df_with_labels, ['Glucose', 'BMI', 'Age'])

    feat_ext = FeatureExtractor()
    X_train, X_test, y_train, y_test = feat_ext.split(df_with_labels, 'Outcome')
    X_train_pca, X_test_pca = feat_ext.pca_analysis(X_train, X_test)

    # Visualize cumulative PCA variance for Diabetes Data
    viz.visualize_cumulative_pca_variance(feat_ext.pca, title="Diabetes Data - PCA")

    viz.visualize_pca(X_train_pca, y_train, title="Diabetes Data - PCA")

    sl = SuperLearner()
    sl.tune_models(X_train_pca, y_train)
    sl.train_final(sl.get_preds(X_train_pca, y_train), y_train)
    print(f"Diabetes Data Accuracy: {sl.check_accuracy(X_test_pca, y_test)}")

    df2 = pd.read_csv("./data_file/Marketing.csv")
    print(df2.describe())
    print("Marketing columns:", df2.columns.tolist())
    enc = Encoder()
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'previous_marketing_outcome']
    df2_enc = enc.encode(df2, cat_cols)
    DataVisualization().visualize_marketing_data_preprocessing(df2_enc)
    df2_clean = prep.preprocess(df2_enc.drop('successful_marketing', axis=1), "Marketing Data")
    df2_clean['successful_marketing'] = df2['successful_marketing']
    DataVisualization().visualize_marketing_data_post_preprocessing(df2_clean.drop('successful_marketing', axis=1))

    X_train2, X_test2, y_train2, y_test2 = feat_ext.split(df2_clean, 'successful_marketing')
    X_train2_pca, X_test2_pca = feat_ext.pca_analysis(X_train2, X_test2)

    # Visualize cumulative PCA variance for Marketing Data
    viz.visualize_cumulative_pca_variance(feat_ext.pca, title="Marketing Data - PCA")

    viz.visualize_pca(X_train2_pca, y_train2, title="Marketing Data - PCA")

    sl.tune_models(X_train2_pca, y_train2)
    sl.train_final(sl.get_preds(X_train2_pca, y_train2), y_train2)
    print(f"Marketing Data Accuracy: {sl.check_accuracy(X_test2_pca, y_test2)}")


if __name__ == "__main__":
    main()
