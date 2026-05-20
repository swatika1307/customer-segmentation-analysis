import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Page Title
st.title("Customer Segmentation Analysis")

# Project Description
st.write("""
This application performs customer segmentation using K-Means clustering.
Upload a customer dataset to begin analysis.
""")

# File Upload
uploaded_file = st.file_uploader(
    "Upload Customer CSV File",
    type=["csv"]
)

# Read Dataset
if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    st.subheader("Dataset Shape")
    st.write(df.shape)
    
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # EDA Section
    st.header("Exploratory Data Analysis")

    # Age Distribution
    st.subheader("Age Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.histplot(df['Age'], bins=20, kde=True, ax=ax)

    ax.set_title("Distribution of Customer Age")

    st.pyplot(fig)


    # Annual Income Distribution
    st.subheader("Annual Income Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.histplot(
        df['Annual Income (k$)'],
        bins=20,
        kde=True,
        ax=ax
    )

    ax.set_title("Annual Income Distribution")

    st.pyplot(fig)


    # Spending Score Distribution
    st.subheader("Spending Score Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.histplot(
        df['Spending Score (1-100)'],
        bins=20,
        kde=True,
        ax=ax
    )

    ax.set_title("Spending Score Distribution")

    st.pyplot(fig)


    # Income vs Spending Scatter Plot
    st.subheader("Annual Income vs Spending Score")

    fig, ax = plt.subplots(figsize=(8,6))

    sns.scatterplot(
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        data=df,
        ax=ax
    )

    ax.set_title("Income vs Spending Score")

    st.pyplot(fig)


    # Correlation Heatmap
    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap='coolwarm',
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    st.pyplot(fig)

    # -----------------------------------
    # K-Means Clustering Section
    # -----------------------------------

    st.header("Customer Segmentation using K-Means")

    # Feature Selection
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    # Elbow Method
    st.subheader("Elbow Method")

    wcss = []

    for i in range(1, 11):

        kmeans = KMeans(
            n_clusters=i,
            init='k-means++',
            random_state=42
        )

        kmeans.fit(X)

        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8,5))

    ax.plot(range(1,11), wcss, marker='o')

    ax.set_title("Elbow Method")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")

    st.pyplot(fig)


    # Cluster Selection
    st.subheader("Select Number of Clusters")

    n_clusters = st.slider(
        "Choose K value",
        min_value=2,
        max_value=10,
        value=5
    )


    # K-Means Model
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        random_state=42
    )

    y_kmeans = kmeans.fit_predict(X)

    df['Cluster'] = y_kmeans


    # Cluster Visualization
    st.subheader("Customer Segments")

    fig, ax = plt.subplots(figsize=(10,7))

    sns.scatterplot(
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        hue='Cluster',
        palette='Set2',
        data=df,
        s=100,
        ax=ax
    )

    ax.scatter(
        kmeans.cluster_centers_[:,0],
        kmeans.cluster_centers_[:,1],
        s=300,
        c='black',
        label='Centroids'
    )

    ax.set_title("Customer Segments")

    st.pyplot(fig)

    # -----------------------------------
    # Business Insights
    # -----------------------------------

    if n_clusters == 5:

        st.header("Business Insights and Recommendations")

        st.subheader("Cluster 0 — Average Customers")

        st.write("""
        - Moderate income and moderate spending behavior.
        - Represents regular retail customers.

        Recommendations:
        - Offer loyalty programs and seasonal discounts.
        - Maintain customer engagement through personalized offers.
        """)


        st.subheader("Cluster 1 — Premium Customers")

        st.write("""
        - High income and high spending customers.
        - High-value customers contributing significantly to business revenue.

        Recommendations:
        - Provide exclusive memberships and premium services.
        - Target with luxury product campaigns.
        """)


        st.subheader("Cluster 2 — High Spending, Low Income Customers")

        st.write("""
        - Lower income but high spending behavior.
        - Represents highly engaged or impulsive buyers.

        Recommendations:
        - Offer cashback and reward-based programs.
        - Promote affordable product bundles.
        """)


        st.subheader("Cluster 3 — High Income, Low Spending Customers")

        st.write("""
        - High earning customers with lower spending behavior.
        - Indicates untapped purchasing potential.

        Recommendations:
        - Use personalized marketing campaigns.
        - Improve customer engagement strategies.
        """)


        st.subheader("Cluster 4 — Low Income, Low Spending Customers")

        st.write("""
        - Budget-conscious customers with lower engagement.

        Recommendations:
        - Focus on value-based products and discounts.
        - Promote budget-friendly offers.
        """)

    else:

        st.info(
            "Business insights are currently optimized for K = 5 clusters based on the Elbow Method analysis."
        )