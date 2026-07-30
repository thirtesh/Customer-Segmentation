import pandas as pd
import sklearn as sk
from pathlib import Path
import preprocess as pp

BASE_DIR = Path(__file__).resolve().parent.parent

df=pp.load_data(BASE_DIR / 'dataset' / 'Mall_Customers.csv')
df=pp.rename_columns(df, {'Annual Income (k$)':'Annual Income', 'Spending Score (1-100)':'Spending Score'})
df=pp.drop_unwanted_columns(df, ['CustomerID', 'Gender', 'Age'])
df=pp.standardize_data(df)

pp.plot_elbow_graph(df, BASE_DIR / 'screenshots' / 'elbow_plot.png')
pp.plot_silhouette_graph(df, BASE_DIR / 'screenshots' / 'silhouette_plot.png')

kmeans=sk.cluster.KMeans(n_clusters=5, random_state=77, n_init=10)
labels=kmeans.fit_predict(df)

pp.plot_segmentation(df, labels, kmeans.cluster_centers_, BASE_DIR / 'screenshots' / 'segmentation_plot.png')

df=pd.DataFrame(df)
df['cluster']=kmeans.labels_
df['segment']=df['cluster'].map(pp.segment_map)

pp.export_processed_data(df, BASE_DIR / 'dataset' / 'Mall_Customers_Processed.csv')
pp.save_model(kmeans, BASE_DIR / 'model' / 'kmeans.joblib')
pp.save_model(pp.scaler, BASE_DIR / 'model' / 'std_scaler.joblib')
