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
import matplotlib.pyplot as plt



# Data Preprocessing Class
class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')  # fill missing with mean

    def show_outliers(self, df, cols, when="Before", name="Data"):
        print(f"Checking outliers for {name} - {when}")
        plt.figure(figsize=(20, 5))
        for i, col in enumerate(cols, 1):
            plt.subplot(1, len(cols), i)
            sns.boxplot(y=df[col], color='blue')  # blue boxes look cool
            plt.title(col)
            plt.xticks([])  # no ticks on x
        plt.suptitle(f"Outliers {when} Cleaning ({name})")
        plt.tight_layout()
        plt.show()

    def clean_outliers(self, df, cols, thresh=3, name="Data"):
        self.show_outliers(df, cols, "Before", name)
        for col in cols:
            z = np.abs((df[col] - df[col].mean()) / df[col].std())  # z-score calc
            df = df[z < thresh]  # keep only non-outliers
        self.show_outliers(df, cols, "After", name)
        return df

    def fill_missing(self, df, cols):
        df[cols] = self.imputer.fit_transform(df[cols])  # fill NaNs
        return df

    def scale_data(self, df, cols):
        df[cols] = self.scaler.fit_transform(df[cols])  # scale it
        return df

    def preprocess(self, df, name="Data"):
        print(f"Starting preprocessing for {name}")
        print("Columns:", df.columns.tolist())
        cols = df.columns
        df = self.fill_missing(df, cols)
        if name == "Medical Data":
            num_cols = cols  # all are numbers here
        else:
            num_cols = ['age', 'last_contact_duration', 'campaign_contacts',
                        'passed_days', 'previous_contacts']
            num_cols = [c for c in num_cols if c in df.columns]  # only valid ones
            print(f"Numeric columns for {name}: {num_cols}")
        df = self.clean_outliers(df, num_cols, name=name)
        df = self.scale_data(df, cols)
        return df


# Label Maker Class
class LabelMaker:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=2, random_state=42)  # 2 clusters

    def make_labels(self, df, feats):
        print("Making labels with K-means")
        clusters = self.kmeans.fit_predict(df[feats])
        df['Cluster'] = clusters
        means = df.groupby('Cluster')['Glucose'].mean()  # check glucose means
        diabetic = means.idxmax()  # higher glucose = diabetes
        df['Outcome'] = df['Cluster'].apply(lambda x: 1 if x == diabetic else 0)
        df = df.drop('Cluster', axis=1)  # don’t need this anymore
        return df


# Feature Extraction Class
class FeatureExtractor:
    def __init__(self):
        self.pca = PCA(n_components=3)  # 3 components

    def split(self, df, target):
        X = df.drop(target, axis=1)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def pca_analysis(self, X_train, X_test):
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)
        return X_train_pca, X_test_pca


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
        preds = []
        for name, model in self.models:
            p = cross_val_predict(model, X, y, cv=5)
            preds.append(p)
        return np.column_stack(preds)

    def train_final(self, X_meta, y):
        grid = GridSearchCV(self.final_model, {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]},
                            cv=5, scoring='accuracy')
        grid.fit(X_meta, y)
        self.final_model = grid.best_estimator_
        print(f"Final model settings: {grid.best_params_}")

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for _, m in self.models])
        return self.final_model.predict(preds)

    def check_accuracy(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)


# Categorical Encoder Class
class Encoder:
    def __init__(self):
        self.encoders = {}

    def encode(self, df, cols):
        for col in cols:
            self.encoders[col] = LabelEncoder()
            df[col] = self.encoders[col].fit_transform(df[col].astype(str))
        return df


# Main stuff
def main():
    # Medical Data
    df = pd.read_csv("./data_file/diabetes_project.csv")
    prep = DataPreprocessor()
    df_clean = prep.preprocess(df, "Medical Data")
    print("Cleaned Medical Data:\n", df_clean.head())

    # Labels
    label_maker = LabelMaker()
    feats = ['Glucose', 'BMI', 'Age']
    df_with_labels = label_maker.make_labels(df_clean, feats)
    print("Data with labels:\n", df_with_labels.head())

    # Features
    feat_ext = FeatureExtractor()
    X_train, X_test, y_train, y_test = feat_ext.split(df_with_labels, 'Outcome')
    X_train_pca, X_test_pca = feat_ext.pca_analysis(X_train, X_test)
    print("PCA shape:", X_train_pca.shape)

    # Super Learner
    sl = SuperLearner()
    sl.tune_models(X_train_pca, y_train)
    meta_X = sl.get_preds(X_train_pca, y_train)
    sl.train_final(meta_X, y_train)
    acc = sl.check_accuracy(X_test_pca, y_test)
    print(f"Medical Data Accuracy: {acc}")

    # Marketing Data
    df2 = pd.read_csv("./data_file/Marketing.csv")
    print("Marketing columns:", df2.columns.tolist())

    enc = Encoder()
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'previous_marketing_outcome']
    df2_enc = enc.encode(df2, cat_cols)

    df2_clean = prep.preprocess(df2_enc.drop('successful_marketing', axis=1), "Marketing Data")
    df2_clean['successful_marketing'] = df2['successful_marketing']

    X_train2, X_test2, y_train2, y_test2 = feat_ext.split(df2_clean, 'successful_marketing')
    X_train2_pca, X_test2_pca = feat_ext.pca_analysis(X_train2, X_test2)

    sl.tune_models(X_train2_pca, y_train2)
    meta_X2 = sl.get_preds(X_train2_pca, y_train2)
    sl.train_final(meta_X2, y_train2)
    acc2 = sl.check_accuracy(X_test2_pca, y_test2)
    print(f"Marketing Data Accuracy: {acc2}")


if __name__ == "__main__":
    main()