from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

def import_raw_data(csv_path):
    """
    Läser in rådata från CSV och gör grundläggande preprocessing.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    
    # Ta bort kolumner med endast ett unikt värde
    df = df.loc[:, df.nunique(dropna=False) > 1]
    
    # Konvertera target: low_bike_demand=0, high_bike_demand=1
    df["increase_stock"] = (df["increase_stock"] == "high_bike_demand").astype(int)
    
    return df

if __name__ == "__main__":
    df = import_raw_data("data/training_data_VT2026.csv")
    print('\n'
        f"Shape of df: {df.shape}\n"
    )
    print("\nAlla kolumner i df:")
    print(df.columns.tolist())

    print("\nHead:")
    print(df.head())

    print("\nStatistik för df:")
    print(df.describe())
