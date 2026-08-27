import src.preprocess as prc
import sklearn as sk
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

if 'initialized' not in st.session_state:
    st.session_state.df=pd.DataFrame()
    st.session_state.data=pd.DataFrame()
    st.session_state.elbow=[]
    st.session_state.silhouette=[]
    st.session_state.recmd_ind=0

df=st.session_state.df
data=st.session_state.data
elbow=st.session_state.elbow
silhouette=st.session_state.silhouette
recmd_ind= st.session_state.recmd_ind

st.title('Customer Segmentation')
dir = st.file_uploader('Choose a csv file', type='csv')

r1=st.number_input('Enter lower range limit for number of segments:', min_value=2, step=1)
r2=st.number_input('Enter upper range limit for number of segments:', min_value=2, step=1)

if st.button('Preview details'):

    if dir is not None:
        df=pd.read_csv(dir)
        df=df.drop_duplicates()
        df=df.dropna()

        try:
            data = df.loc[:, ['AnnualIncome', 'SpendingScore']]
        except:
            st.error('Make sure columns \'AnnualIncome\' and \'SpendingScore\' are present')

        df=prc.standardize_data(data)

        for k in range(r1,r2+1):
            kmeans=sk.cluster.KMeans(n_clusters=k, random_state=77, n_init=10)
            kmeans.fit(df)
            elbow.append(kmeans.inertia_)

        temp=[]
        for i in range(len(elbow)-1):
            temp.append(abs(elbow[i]-elbow[i+1]))

        temp2=[]
        for i in range(len(temp)-1):
            temp2.append(abs(temp[i]-temp[i+1]))

        ind=temp2.index(max(temp2))
        elbow_point = ind+1+r1

        for k in range(r1,r2+1):
            kmeans=sk.cluster.KMeans(n_clusters=k, random_state=77, n_init=10)
            pred=kmeans.fit_predict(df)
            score=sk.metrics.silhouette_score(df,pred)
            silhouette.append(score)

        recmd_ind=silhouette.index(max(silhouette[elbow_point-1:elbow_point+1]))
        
        st.success('The dataset showed best results for:\n ' \
        f'Number of clusters:{recmd_ind}\n' \
        f'Elbow score:{elbow[recmd_ind]}\n' \
        f'Silhouette Score:{silhouette[recmd_ind]}\n')

k = st.number_input('Enter k value(enter 0 to use recommended clustering):')

if st.button('Get Clustered Dataset'):

    if k!=0:
        recmd_ind=k

    kmeans=sk.cluster.KMeans(n_clusters=recmd_ind, random_state=77, n_init=10)
    labels=kmeans.fit_predict(df)

    centers=kmeans.cluster_centers_
    X=df

    plt.figure(figsize=(8,5))
    plt.plot(range(1,10), elbow, marker='o')
    plt.title('Elbow method plot')
    plt.grid(True)
    st.pyplot(plt.gcf())

    plt.figure(figsize=(8,5))
    plt.plot(range(2,10), silhouette, marker='o')
    plt.title('Silhouette method plot')
    plt.grid(True)
    st.pyplot(plt.gcf())

    plt.figure(figsize=(8,5))
    plt.scatter(df[:, 0], df[:, 1], c=labels, cmap='viridis')
    plt.scatter(centers[:,0], centers[:,1], c='red', label='Centroids')
    plt.title('Customer Segmentation')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.grid(True)
    st.pyplot(plt.gcf())

    data['cluster']=kmeans.labels_


st.session_state.df=df
st.session_state.data=data
st.session_state.elbow=elbow
st.session_state.silhouette=silhouette
st.session_state.recmd_ind=recmd_ind