import streamlit as st
import pandas as pd

from src.data_cleaning import clean_data
from src.eda import (
    plot_age_distribution,
    plot_income_distribution,
    plot_spending_distribution,
    plot_gender_distribution,
    plot_income_vs_spending,
    plot_correlation_heatmap
)

from src.clustering import (
    select_features,
    calculate_wcss,
    perform_clustering,
    add_cluster_labels,
    generate_cluster_summary
)

from src.visualization import (
    plot_elbow_method,
    plot_clusters
)


# -----------------------------------
# Page Title
# -----------------------------------

st.title("Customer Segmentation Analysis")


st.write("""
This application performs customer segmentation using K-Means clustering.
Upload a customer dataset to analyze customer behavior and identify distinct customer groups.
""")


# -----------------------------------
# File Upload
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload Customer CSV File",
    type=["csv"]
)


# -----------------------------------
# Main App
# -----------------------------------

if uploaded_file is not None:

    # Load Dataset
    df = pd.read_csv(uploaded_file)

    # Clean Dataset
    df = clean_data(df)

    # -----------------------------------
    # Dataset Preview
    # -----------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(df.head())


    # -----------------------------------
    # Dataset Shape
    # -----------------------------------

    st.subheader("Dataset Shape")

    st.write(df.shape)


    # -----------------------------------
    # Summary Statistics
    # -----------------------------------

    st.subheader("Summary Statistics")

    st.write(df.describe())


    # -----------------------------------
    # Exploratory Data Analysis
    # -----------------------------------

    st.header("Exploratory Data Analysis")

    st.subheader("Age Distribution")

    st.pyplot(plot_age_distribution(df))


    st.subheader("Annual Income Distribution")

    st.pyplot(plot_income_distribution(df))


    st.subheader("Spending Score Distribution")

    st.pyplot(plot_spending_distribution(df))


    st.subheader("Gender Distribution")

    st.pyplot(plot_gender_distribution(df))


    st.subheader("Annual Income vs Spending Score")

    st.pyplot(plot_income_vs_spending(df))


    st.subheader("Correlation Heatmap")

    st.pyplot(plot_correlation_heatmap(df))


    # -----------------------------------
    # K-Means Clustering
    # -----------------------------------

    st.header("Customer Segmentation using K-Means")


    # Feature Selection
    X = select_features(df)


    # -----------------------------------
    # Elbow Method
    # -----------------------------------

    st.subheader("Elbow Method")

    wcss = calculate_wcss(X)

    st.pyplot(plot_elbow_method(wcss))


    # -----------------------------------
    # Cluster Selection
    # -----------------------------------

    st.subheader("Select Number of Clusters")

    n_clusters = st.slider(
        "Choose K value",
        min_value=2,
        max_value=10,
        value=5
    )


    # -----------------------------------
    # Perform Clustering
    # -----------------------------------

    kmeans, labels = perform_clustering(
        X,
        n_clusters
    )

    df = add_cluster_labels(df, labels)


    # -----------------------------------
    # Cluster Visualization
    # -----------------------------------

    st.subheader("Customer Segments")

    st.pyplot(plot_clusters(df, kmeans))


    # -----------------------------------
    # Cluster Summary
    # -----------------------------------

    st.subheader("Cluster Summary")

    cluster_summary = generate_cluster_summary(df)

    st.dataframe(cluster_summary)


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


# -----------------------------------
# Footer
# -----------------------------------

st.markdown("---")

st.write(
    "Customer Segmentation Analysis Dashboard using Python, Scikit-learn, and Streamlit"
)