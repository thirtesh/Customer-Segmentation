import sklearn as sk
import joblib as jb
from pathlib import Path
import preprocess as pp

BASE_DIR = Path(__file__).resolve().parent.parent

df=pp.load_data(BASE_DIR / 'dataset' / 'Mall_Customers_Processed.csv')

pp.plot_elbow_graph(df.iloc[:,0:2], BASE_DIR / 'screenshots' / 'elbow_plot.png')

pp.plot_silhouette_graph(df.iloc[:,0:2], BASE_DIR / 'screenshots' / 'silhouette_plot.png')

scaler=jb.load(BASE_DIR/'model'/'std_scaler.joblib')

kmeans=jb.load(BASE_DIR/'model'/'kmeans.joblib')
labels=kmeans.fit_predict(df.iloc[:,0:2])

print(f'The inertia of the clustered dataset(n_clusters=5):{kmeans.inertia_}\nThe silhouette score of the clustered dataset:{sk.metrics.silhouette_score(df.iloc[:,0:2], labels)}')

pred=pp.segment_map[kmeans.predict(scaler.transform([[40,70]]))[0]]
print(f'A person with a salary of 40k$ and spending score of 70 is "{pred}"')
