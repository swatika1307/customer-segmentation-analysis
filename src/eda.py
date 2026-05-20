import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style='whitegrid')


def plot_age_distribution(df):

    fig, ax = plt.subplots(figsize=(8,5))

    sns.histplot(
        df['Age'],
        bins=20,
        kde=True,
        ax=ax
    )

    ax.set_title("Distribution of Customer Age")

    return fig


def plot_income_distribution(df):

    fig, ax = plt.subplots(figsize=(8,5))

    sns.histplot(
        df['Annual Income (k$)'],
        bins=20,
        kde=True,
        ax=ax
    )

    ax.set_title("Annual Income Distribution")

    return fig


def plot_spending_distribution(df):

    fig, ax = plt.subplots(figsize=(8,5))

    sns.histplot(
        df['Spending Score (1-100)'],
        bins=20,
        kde=True,
        ax=ax
    )

    ax.set_title("Spending Score Distribution")

    return fig


def plot_gender_distribution(df):

    fig, ax = plt.subplots(figsize=(6,4))

    sns.countplot(
        x='Gender',
        data=df,
        ax=ax
    )

    ax.set_title("Gender Distribution")

    return fig


def plot_income_vs_spending(df):

    fig, ax = plt.subplots(figsize=(8,6))

    sns.scatterplot(
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        data=df,
        ax=ax
    )

    ax.set_title("Income vs Spending Score")

    return fig


def plot_correlation_heatmap(df):

    fig, ax = plt.subplots(figsize=(8,5))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap='coolwarm',
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    return fig