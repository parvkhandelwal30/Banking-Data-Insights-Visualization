import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset 
df = pd.read_csv(r"D:\Desktop\programming\banking_data.csv")

# Function to visualize categorical distributions and display numeric stats
def plot_categorical_distribution(column, title):
    if column in df.columns:
        # Plotting the distribution
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=column, palette='viridis', order=df[column].value_counts().index)
        plt.title(title)
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # Displaying basic stats for categorical variables
        print(f"\n{title} - Stats:\n")
        print(df[column].value_counts())
        print(f"\nMissing Values: {df[column].isnull().sum()}")
        print("-" * 50, "\n")
    else:
        print(f"Column '{column}' not found in the dataset.")

# Function to visualize numeric distributions and display numeric stats
def plot_numeric_distribution(column, title):
    if column in df.columns:
        # Plotting the distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x=column, kde=True, color='blue', bins=30)
        plt.title(title)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

        # Displaying basic stats for numeric variables
        print(f"\n{title} - Stats:\n")
        print(f"Mean: {df[column].mean():.2f}")
        print(f"Median: {df[column].median():.2f}")
        print(f"Standard Deviation: {df[column].std():.2f}")
        print(f"Min: {df[column].min()}")
        print(f"Max: {df[column].max()}")
        print(f"25th Percentile: {df[column].quantile(0.25):.2f}")
        print(f"75th Percentile: {df[column].quantile(0.75):.2f}")
        print(f"Missing Values: {df[column].isnull().sum()}")
        print("-" * 50, "\n")
    else:
        print(f"Column '{column}' not found in the dataset.")

# Questions and visualizations
questions = [
    ("age", "Distribution of Age"),
    ("job", "Job Type Distribution"),
    ("marital", "Marital Status Distribution"),
    ("education", "Education Level Distribution"),
    ("default", "Proportion of Clients with Credit Default"),
    ("balance", "Distribution of Average Yearly Balance"),
    ("housing", "Clients with Housing Loans"),
    ("loan", "Clients with Personal Loans"),
    ("contact", "Communication Types Used During Campaigns"),
    ("day", "Distribution of Last Contact Day"),
    ("month", "Last Contact Month Variation"),
    ("duration", "Distribution of Last Contact Duration"),
    ("campaign", "Contacts Performed During Campaign"),
    ("pdays", "Days Since Last Contact"),
    ("previous", "Contacts Before Current Campaign"),
    ("poutcome", "Outcomes of Previous Campaigns"),
    ("y", "Term Deposit Subscription Distribution")
]

# Iterate through the questions and display results
for col, title in questions:
    if df[col].dtype == 'object':  # Check if column is categorical
        plot_categorical_distribution(col, title)
    else:  # Otherwise, it's numeric
        plot_numeric_distribution(col, title)

# Correlation matrix for numeric attributes
if not df.empty:
    numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
    if not numeric_df.empty:
        print("\nCorrelation Matrix:\n")
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()
    else:
        print("No numeric columns available for correlation.")
else:
    print("Dataset is empty. Unable to compute correlations.")
