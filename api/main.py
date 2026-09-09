import numpy as np
import joblib as jb
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel
import sklearn as sk

def standardize_data(data):
    try:
        scaler = sk.preprocessing.StandardScaler()
        data = scaler.fit_transform(data)
        return data
    except Exception as e:
         raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, detail=f"Error standardizing data: {e}")
 


app= FastAPI()

class PreviewData(BaseModel):
    rr1:int
    rr2:int

@app.post('/preview')
def preview(formdata:str=Form(...),  file: UploadFile=File(...)):

    global r1,r2,df,data,elbow,silhouette,k_values,recommended_k

    formdata= PreviewData.model_validate_json(formdata)
    r1=formdata.rr1
    r2=formdata.rr2

    if file is None:
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, detail="Please upload a CSV file first.")

    if r2 <= r1:
         raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, detail="Upper range limit must be greater than the lower range limit.")

    if r2 - r1 < 2:
         raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, detail="Please provide a range containing at least 3 K values.")

    try:
        df = pd.read_csv(file.file)
    except Exception as e:
         raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, detail=f"Could not read the CSV file: {e}")


    df = df.drop_duplicates()
    data = df.dropna()

    try:
        df = data.loc[:, ["AnnualIncome", "SpendingScore"]].copy()
    except KeyError:
         raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, detail="Make sure columns 'AnnualIncome' and 'SpendingScore' are present.")

    if len(df) < r2:
         raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, detail= f"The dataset must contain at least {r2} rows to test up to K={r2}.")

    X = standardize_data(df)

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
    df = X

    return {'recommended_k':recommended_k, 'elbow_k':elbow_k, 'silhouette_k':silhouette[best_k_index]}


@app.post('/predict')
async def predict(k:int=Form(...)):

    if k != 0 and not r1 <= k <= r2:
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, detail=f"K must be between {r1} and {r2}.")

    selected_k = recommended_k if k == 0 else int(k)

    if selected_k < 1:
        raise HTTPException(status_code = status.HTTP_400_BAD_REQUEST, detail="No valid recommended K is available. Please preview the dataset first.")

    kmeans = sk.cluster.KMeans(n_clusters=selected_k, random_state=77, n_init=10)

    labels = kmeans.fit_predict(df)
    centers = kmeans.cluster_centers_

    return {
            'data':data.to_dict(orient='records'),
            'df':df.tolist(),
            'k_values':k_values,
            'elbow':elbow,
            'silhouette':silhouette,
            'labels':labels.tolist(),
            'centers':centers.tolist(),
            'selected_k':selected_k
            }


    

