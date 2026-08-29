import src.preprocess as prc
import sklearn as sk
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.df = pd.DataFrame()
    st.session_state.data = pd.DataFrame()
    st.session_state.elbow = []
    st.session_state.silhouette = []
    st.session_state.k_values = []
    st.session_state.recommended_k = 0
    st.session_state.analysis_done = False

df = st.session_state.df
data = st.session_state.data
elbow = st.session_state.elbow
silhouette = st.session_state.silhouette
k_values = st.session_state.k_values
recommended_k = st.session_state.recommended_k


st.title("Customer Segmentation")

uploaded_file = st.file_uploader("Choose a csv file", type="csv")

r1 = st.number_input("Enter lower range limit for number of segments:", min_value=2, step=1)

r2 = st.number_input("Enter upper range limit for number of segments:", min_value=2, step=1)


if st.button("Preview details"):

    if uploaded_file is None:
        st.error("Please upload a CSV file first.")
        st.stop()

    if r2 <= r1:
        st.error("Upper range limit must be greater than the lower range limit.")
        st.stop()

    if r2 - r1 < 2:
        st.error("Please provide a range containing at least 3 K values.")
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the CSV file: {e}")
        st.stop()

    df = df.drop_duplicates()
    data = df.dropna()

    try:
        df = data.loc[:, ["AnnualIncome", "SpendingScore"]].copy()
    except KeyError:
        st.error(
            "Make sure columns 'AnnualIncome' and 'SpendingScore' are present."
        )
        st.stop()

    if len(df) < r2:
        st.error(
            f"The dataset must contain at least {r2} rows to test up to K={r2}."
        )
        st.stop()

    X = prc.standardize_data(df)

    elbow = []
    silhouette = []
    k_values = list(range(r1, r2 + 1))

    for k in k_values:
        kmeans = sk.cluster.KMeans(n_clusters=k, random_state=77, n_init=10)
        kmeans.fit(X)
        elbow.append(kmeans.inertia_)

    temp = [
        abs(elbow[i] - elbow[i + 1])
        for i in range(len(elbow) - 1)]

    temp2 = [
        abs(temp[i] - temp[i + 1])
        for i in range(len(temp) - 1)]

    elbow_index = temp2.index(max(temp2)) + 1
    elbow_k = k_values[elbow_index]


    for k in k_values:
        kmeans = sk.cluster.KMeans(n_clusters=k, random_state=77, n_init=10)
        pred = kmeans.fit_predict(X)
        score = sk.metrics.silhouette_score(X, pred)
        silhouette.append(score)

    if r1 <= k <= r2:
        ks = [elbow_k - 1, elbow_k, elbow_k + 1]

    k_indices = [
        k_values.index(k) for k in ks]

    best_k_index = max(k_indices, key=lambda i: silhouette[i])

    recommended_k = k_values[best_k_index]

    st.session_state.df = X
    st.session_state.data = data
    st.session_state.elbow = elbow
    st.session_state.silhouette = silhouette
    st.session_state.k_values = k_values
    st.session_state.recommended_k = recommended_k
    st.session_state.analysis_done = True

    df = X
    st.session_state.df = pd.DataFrame(df)

    st.success(
        f"The dataset showed best results for:\n\n"
        f"Recommended number of clusters: {recommended_k}\n\n"
        f"Elbow point: {elbow_k}\n\n"
        f"Silhouette Score: {silhouette[best_k_index]:.4f}"
    )


k = st.number_input("Enter k value (enter 0 to use recommended clustering):",min_value=0,step=1)


if st.button("Get Clustered Dataset"):

    if not st.session_state.analysis_done:
        st.error("Please click 'Preview details' and analyze the dataset first.")
        st.stop()
        
    if k != 0 and not r1 <= k <= r2:
        st.error(f"K must be between {r1} and {r2}.")
        st.stop()

    selected_k = recommended_k if k == 0 else int(k)

    if selected_k < 1:
        st.error("No valid recommended K is available. Please preview the dataset first.")
        st.stop()

    kmeans = sk.cluster.KMeans(n_clusters=selected_k, random_state=77, n_init=10)

    labels = kmeans.fit_predict(df)
    centers = kmeans.cluster_centers_

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(k_values, elbow, marker="o")
    ax1.set_title("Elbow Method Plot")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia / WCSS")
    ax1.grid(True)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(k_values, silhouette, marker="o")
    ax2.set_title("Silhouette Method Plot")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.grid(True)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.scatter(df.iloc[:, 0],  df.iloc[:, 1],  c=labels,  cmap="viridis")
    ax3.scatter(centers[:, 0],  centers[:, 1],  c="red",  label="Centroids")
    ax3.set_title(f"Customer Segmentation (K={selected_k})")
    ax3.set_xlabel("Annual Income (standardized)")
    ax3.set_ylabel("Spending Score (standardized)")
    ax3.grid(True)
    ax3.legend()
    st.pyplot(fig3)

    result = data.copy()
    result["cluster"] = labels

    st.subheader("Clustered Dataset")
    st.dataframe(result)

    csv = result.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Clustered CSV",
        data=csv,
        file_name="clustered_customers.csv",
        mime="text/csv",
    )
