import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler

ARFF_PATH = "water-treatment_imputed.arff"
JSON_OUTPUT_PATH = "water-treatment_normalized.json"

#load the arff file (already imputed: 38 numeric attributes, no missing values)
raw_data, meta = arff.loadarff(ARFF_PATH)
df = pd.DataFrame(raw_data)

#normalize the features (zero mean, unit variance) for use in K-Means
scaler = StandardScaler()
normalized_values = scaler.fit_transform(df)
normalized_df = pd.DataFrame(normalized_values, columns=df.columns)

#save the normalized dataset as JSON records for the clustering script
normalized_df.to_json(JSON_OUTPUT_PATH, orient="records", indent=2)

print(f"Normalized {normalized_df.shape[0]} instances x {normalized_df.shape[1]} features")
print(f"Saved normalized data to {JSON_OUTPUT_PATH}")
