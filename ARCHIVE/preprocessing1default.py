from cProfile import label
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

import numpy as np
 
with open("fault_detection_dataset.json") as f:
    dataset = json.load(f)

meta = dataset["meta"]
data = pd.DataFrame(dataset["records"])

data.head()

#convert timestamp to datetime´
data['timestamp'] = pd.to_datetime(data['timestamp'])

#remove duplicates, used the {https://www.linkedin.com/pulse/data-cleaning-python-handling-duplicates-pandas-bennett-alexander-ay7xe/}
data_noduplicates = data.drop_duplicates()
data_noduplicates.head()
print(f'The number of duplicates removed: {len(data) - len(data_noduplicates)}   ')


OUTPUT_PATH = Path(__file__).parent / "fault_detection_checkrelevant_dataset.json"
data_noduplicates.to_json(OUTPUT_PATH, orient="records", date_format="iso", indent=2)

#missing values {https://www.kaggle.com/code/alexisbcook/handling-missing-values}
def analyze_data_quality(df):
    missing_values_count = df.isnull().sum()
    nan_values_count = df.isna().sum()
    zero_values_count = (df == 0).sum()
    print(f'Missing values count:\n{missing_values_count}')
    print(f'NaN values count:\n{nan_values_count}')
    print(f'Zero values count:\n{zero_values_count}')

    print(f'The number of missing values: {missing_values_count.sum()}')
    print(f'The number of NaN values: {nan_values_count.sum()}')
    print(f'The number of zero values: {zero_values_count.sum()}')

    total_cells = np.prod(df.shape)
    total_missing = missing_values_count.sum()
    print(f'The percentage of missing values: {(total_missing / total_cells) * 100:.2f}%')
    print(f'The percentage of NaN values: {(nan_values_count.sum() / total_cells) * 100:.2f}%')
    print(f'The percentage of zero values: {(zero_values_count.sum() / total_cells) * 100:.2f}%')

    return missing_values_count, nan_values_count, zero_values_count

missing_values_count, nan_values_count, zero_values_count = analyze_data_quality(data_noduplicates)

#handle missing values by interpolation
data_interpolated = data_noduplicates.sort_values(by='timestamp').copy()
data_interpolated[['sensor_1', 'sensor_2', 'sensor_3']] = data_interpolated[['sensor_1', 'sensor_2', 'sensor_3']].interpolate(method='linear', limit_direction='both')

#output_path_interpolated = Path(__file__).parent / "fault_detection_interpolated_dataset.json"
#data_interpolated.to_json(output_path_interpolated, orient="records", date_format="iso", indent=2)

missing_values_count, nan_values_count, zero_values_count = analyze_data_quality(data_interpolated)

state = data_interpolated['state']

