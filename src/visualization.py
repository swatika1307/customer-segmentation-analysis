import matplotlib.pyplot as plt
import seaborn as sns


def plot_elbow_method(wcss):

    fig, ax = plt.subplots(figsize=(8,5))

    ax.plot(range(1,11), wcss, marker='o')

    ax.set_title("Elbow Method")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")

    return fig


def plot_clusters(df, kmeans):

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

    return fig