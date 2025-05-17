import pandas as pd

raw_csv = 'dataexport_20250514T175520.csv'
df = pd.read_csv(
    raw_csv,
    skiprows=10,
    header=None,
    names=['timestamp', 'temperature', 'solar_radiation', 'precipitation', 'cloudiness'],
    low_memory=False
)

for col in ['temperature', 'solar_radiation', 'precipitation', 'cloudiness']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['date'] = pd.to_datetime(df['timestamp'], format='%Y%m%dT%H%M', errors='coerce')
df.drop(columns=['timestamp'], inplace=True)

df = df.dropna(subset=['date', 'temperature'])

df_daily = (
    df
    .set_index('date')
    .resample('D')
    .mean()
    .reset_index()
)

daily_csv = 'daily_weather.csv'
df_daily.to_csv(daily_csv, index=False)
print(f"Saved daily CSV to {daily_csv}")