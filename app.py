import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt

st.title("Customer Segmentation")

uploaded_file = st.file_uploader("Choose a csv file", type="csv")

r1 = st.number_input("Enter lower range limit for number of segments:", min_value=2, step=1)

r2 = st.number_input("Enter upper range limit for number of segments:", min_value=2, step=1)


if st.button("Preview details"):
    
    file={
            'file': uploaded_file
            }

    data={
            'formdata':json.dumps({
                'rr1':r1,
                'rr2':r2
                })
            }
   
    response = requests.post("http://0.0.0.0:8000/preview", 
                             data=data,
                             files=file)
        
    if response.ok:
        result = response.json()
        
        recommended_k=result['recommended_k']
        elbow_k=result['elbow_k']
        silhouette_k=result['silhouette_k']

        st.success(
                f"The dataset showed best results for:\n\n"
            f"Recommended number of clusters: {recommended_k}\n\n"
            f"Elbow point: {elbow_k}\n\n"
            f"Silhouette Score: {silhouette_k:.4f}"
        )
    else:
        st.error(f'API error: {response.status_code}')
        st.write(response.text)

k = st.number_input("Enter k value (enter 0 to use recommended clustering):",min_value=0,step=1)


if st.button("Get Clustered Dataset"):      
    
    d = {
            'k':k
            }
    response=requests.post("http://0.0.0.0:8000/predict", 
                           data=d)
    
    if response.ok:
        result = response.json()
        data = pd.DataFrame(result['data'])
        df = pd.DataFrame(result['df'])
        labels =  pd.DataFrame(result['labels'])
        centers = pd.DataFrame(result['centers'])
        k_values = result['k_values']
        elbow = result['elbow']
        silhouette = result['silhouette']
        selected_k=result['selected_k']


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
        ax3.scatter(centers.iloc[:, 0],  centers.iloc[:, 1],  c="red",  label="Centroids")
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

    else:
        st.error(f'API error: {response.status_code}')
        st.write(response.text)

