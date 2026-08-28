import pandas as pd

INPUT_PATH = "fault_detection.json"
OUTPUT_PATH = "fault_detection_interpolated.json"
SENSOR_COLUMNS = ["sensor_1", "sensor_2", "sensor_3"]


def main():
    df = pd.read_json(INPUT_PATH, convert_dates=False)

    for col in SENSOR_COLUMNS:
        df[col] = df[col].mask(df[col] == 0.0)
        df[col] = df[col].interpolate(method="linear", limit_direction="both")

    df.to_json(OUTPUT_PATH, orient="records", indent=2)
    print(f"Saved interpolated data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
