from sklearn.cluster import KMeans


def select_features(df):
    """
    Select features for clustering.
    """

    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    return X


def calculate_wcss(X):
    """
    Calculate WCSS values for Elbow Method.
    """

    wcss = []

    for i in range(1, 11):

        kmeans = KMeans(
            n_clusters=i,
            init='k-means++',
            random_state=42
        )

        kmeans.fit(X)

        wcss.append(kmeans.inertia_)

    return wcss


def perform_clustering(X, n_clusters=5):
    """
    Perform K-Means clustering.
    """

    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        random_state=42
    )

    labels = kmeans.fit_predict(X)

    return kmeans, labels


def add_cluster_labels(df, labels):
    """
    Add cluster labels to dataframe.
    """

    df['Cluster'] = labels

    return df


def generate_cluster_summary(df):
    """
    Generate cluster summary statistics.
    """

    cluster_summary = df.groupby('Cluster')[
        ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    ].mean()

    return cluster_summary