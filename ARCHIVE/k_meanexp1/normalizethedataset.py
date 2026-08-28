import pandas as pd
from sklearn.preprocessing import StandardScaler

dataset_path = "winsorized_dataset.json"
JSON_OUTPUT_PATH = "winsorized_dataset.json"
SENSOR_COLUMNS = ["sensor_1", "sensor_2", "sensor_3"]


df = pd.read_json(dataset_path, convert_dates=False)

#normalize the features (zero mean, unit variance) for use in K-Means
scaler = StandardScaler()
normalized_values = scaler.fit_transform(df[SENSOR_COLUMNS])
normalized_df = df.copy()
normalized_df[SENSOR_COLUMNS] = normalized_values

#save the normalized dataset as JSON records for the clustering script
normalized_df.to_json(JSON_OUTPUT_PATH, orient="records", indent=2)

print(f"Normalized {normalized_df.shape[0]} instances x {len(SENSOR_COLUMNS)} features")
print(f"Saved normalized data to {JSON_OUTPUT_PATH}")
