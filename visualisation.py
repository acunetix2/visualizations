# iris_analysis.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def load_dataset():
    try:
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        print("✅ Dataset loaded successfully.\n")
        return df
    except Exception as e:
        print("❌ Error loading dataset:", e)
        return None

def explore_dataset(df):
    print("📌 First 5 rows of the dataset:")
    print(df.head())

    print("\n📊 Dataset Information:")
    print(df.info())

    print("\n🔍 Missing values in dataset:")
    print(df.isnull().sum())

def analyze_dataset(df):
    print("\n📈 Descriptive Statistics:")
    print(df.describe())

    grouped_means = df.groupby('species').mean()
    print("\n📊 Mean of Features by Species:")
    print(grouped_means)

    print("\n🔎 Insight: Iris-virginica generally has the highest values for most features.\n")

def visualize_dataset(df):
    sns.set(style="whitegrid")

    # Line Chart
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length", color='blue')
    plt.title("Line Chart - Sepal Length over Index")
    plt.xlabel("Index")
    plt.ylabel("Sepal Length (cm)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Bar Chart
    plt.figure(figsize=(8, 5))
    sns.barplot(x="species", y="petal length (cm)", data=df, ci=None, palette="muted")
    plt.title("Bar Chart - Average Petal Length by Species")
    plt.xlabel("Species")
    plt.ylabel("Petal Length (cm)")
    plt.tight_layout()
    plt.show()

    # Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(df["sepal length (cm)"], bins=20, kde=True, color='green')
    plt.title("Histogram - Sepal Length Distribution")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Scatter Plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df, palette="deep")
    plt.title("Scatter Plot - Sepal vs Petal Length")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.legend(title="Species")
    plt.tight_layout()
    plt.show()

def main():
    df = load_dataset()
    if df is not None:
        explore_dataset(df)
        analyze_dataset(df)
        visualize_dataset(df)

if __name__ == "__main__":
    main()
