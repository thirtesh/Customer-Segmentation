import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def rename_columns(data, columns_mapping):
    try:
        data.rename(columns=columns_mapping, inplace=True)
        return data
    except Exception as e:
        print(f"Error renaming columns: {e}")
        return None
    
def drop_unwanted_columns(data, columns_to_drop):
    try:
        data.drop(columns=columns_to_drop, inplace=True)
        return data
    except Exception as e:
        print(f"Error dropping columns: {e}")
        return None
    
def standardize_data(data):
    try:
        global scaler
        scaler = sk.preprocessing.StandardScaler()
        data = scaler.fit_transform(data)
        return data
    except Exception as e:
        print(f"Error standardizing data: {e}")
        return None
    
def plot_elbow_graph(df,file):
    elbow=[]

    for k in range(1,10):
        kmeans=sk.cluster.KMeans(n_clusters=k, random_state=77, n_init=10)
        kmeans.fit(df)
        elbow.append(kmeans.inertia_)

    plt.figure(figsize=(8,5))
    plt.plot(range(1,10), elbow, marker='o')
    plt.title('Elbow method plot')
    plt.grid(True)
    plt.savefig(file)
    plt.show()

def plot_silhouette_graph(df,file):
    silhouette=[]
    for k in range(2,10):
        kmeans=sk.cluster.KMeans(n_clusters=k, random_state=77, n_init=10)
        labels=kmeans.fit_predict(df)
        silhouette.append(sk.metrics.silhouette_score(df, labels))

    plt.figure(figsize=(8,5))
    plt.plot(range(2,10), silhouette, marker='o')
    plt.title('Silhouette method plot')
    plt.grid(True)
    plt.savefig(file)
    plt.show()

def plot_segmentation(df, labels, centers, file):
    plt.figure(figsize=(8,5))
    plt.scatter(df[:, 0], df[:, 1], c=labels, cmap='viridis')
    plt.scatter(centers[:,0], centers[:,1], c='red', label='Centroids')
    plt.title('Customer Segmentation')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.grid(True)
    plt.savefig(file)
    plt.show()

def export_processed_data(data, file_path):
    try:
        data.to_csv(file_path, index=False)
        print(f"Preprocessed data exported to {file_path}")
    except Exception as e:
        print(f"Error exporting preprocessed data: {e}")

def save_model(model, file_path):
    try:
        import joblib
        joblib.dump(model, file_path)
        print(f"Model saved to {file_path}")
    except Exception as e:
        print(f"Error saving model: {e}")