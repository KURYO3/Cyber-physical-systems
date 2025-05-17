from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io, base64
from astropy.time import Time
from astropy.coordinates import get_sun, EarthLocation
import astropy.units as u

app = Flask(__name__)

MODEL_PATH       = 'weather_multioutput.pkl'
CLIMATOLOGY_PATH = 'climatology.pkl'
CSV_PATH         = 'daily_weather_with_daylength.csv'
LAT, LON         = 47.54694, 7.56892
LOCATION         = EarthLocation(lat=LAT * u.deg, lon=LON * u.deg)

def compute_day_length(date):
    t = Time(f"{date.date()} 12:00:00")
    dec = get_sun(t).dec.rad
    phi = LOCATION.lat.rad
    cosw = -np.tan(phi) * np.tan(dec)
    cosw = np.clip(cosw, -1, 1)
    omega0 = np.arccos(cosw)
    return 2 * omega0 * 24 / (2*np.pi)

def build_feature_row(df_hist, next_date, climatology):
    doy = next_date.dayofyear
    dl  = compute_day_length(next_date)

    row = {
        'dayofyear': doy,
        'month':     next_date.month,
        'weekday':   next_date.weekday(),
        'sin_doy':   np.sin(2 * np.pi * doy / 365),
        'cos_doy':   np.cos(2 * np.pi * doy / 365),
        'day_length_hours': dl,
        'dl_x_sin': dl * np.sin(2 * np.pi * doy / 365),
        'dl_x_cos': dl * np.cos(2 * np.pi * doy / 365),
    }

    bins   = [0, 8, 12, 16, 24]
    labels = ['deep_winter','winter_spring','spring_summer','long_summer']
    season = pd.cut([dl], bins=bins, labels=labels)[0]
    for lab in labels:
        row[f'season_dl_{lab}'] = 1 if lab == season else 0

    df_hist['temp_residual'] = df_hist['temperature'] - df_hist['date'].dt.dayofyear.map(climatology)
    df_hist['temp_roll3']  = df_hist['temp_residual'].shift(1).rolling(3,  min_periods=1).mean()
    df_hist['temp_roll7']  = df_hist['temp_residual'].shift(1).rolling(7,  min_periods=1).mean()
    df_hist['temp_roll14'] = df_hist['temp_residual'].shift(1).rolling(14, min_periods=1).mean()
    df_hist['temp_roll28'] = df_hist['temp_residual'].shift(1).rolling(28, min_periods=1).mean()
    last = df_hist.iloc[-1]
    row['temp_roll3']  = last['temp_roll3']
    row['temp_roll7']  = last['temp_roll7']
    row['temp_roll14'] = last['temp_roll14']
    row['temp_roll28'] = last['temp_roll28']
    row['temp_trend7'] = last['temp_residual'] - last['temp_roll7']

    for col in ['solar_radiation','precipitation','cloudiness']:
        df_hist[f'{col}_roll7'] = df_hist[col].shift(1).rolling(7, min_periods=1).mean()
        row[f'{col}_roll7'] = df_hist[f'{col}_roll7'].iloc[-1]

    for lag in range(1, 8):
        ld = next_date - pd.Timedelta(days=lag)
        if ld in df_hist['date'].values:
            sub = df_hist[df_hist['date']==ld].iloc[0]
            row[f'temp_residual_lag{lag}']    = sub['temp_residual']
            row[f'cloudiness_lag{lag}']       = sub['cloudiness']
            row[f'precipitation_lag{lag}']    = sub['precipitation']
            row[f'solar_radiation_lag{lag}']  = sub['solar_radiation']
            row[f'day_length_hours_lag{lag}'] = sub['day_length_hours']
        else:
            row[f'temp_residual_lag{lag}']    = 0.0
            row[f'cloudiness_lag{lag}']       = df_hist['cloudiness'].mean()
            row[f'precipitation_lag{lag}']    = df_hist['precipitation'].mean()
            row[f'solar_radiation_lag{lag}']  = df_hist['solar_radiation'].mean()
            row[f'day_length_hours_lag{lag}'] = dl

    ld14 = next_date - pd.Timedelta(days=14)
    if ld14 in df_hist['date'].values:
        sub = df_hist[df_hist['date']==ld14].iloc[0]
        row['temp_lag_14'] = sub['temp_residual']
    else:
        row['temp_lag_14'] = 0.0

    return row

def recursive_forecast(n_days, end_date):
    hist = historical_df[historical_df['date'] <= end_date].copy()
    forecasts = []
    for _ in range(n_days):
        next_date = hist['date'].max() + pd.Timedelta(days=1)
        feat = build_feature_row(hist, next_date, climatology)
        y_pred = model.predict(pd.DataFrame([feat]))[0]
        temp = y_pred[0] * scale_factor + climatology[next_date.dayofyear]
        forecasts.append({'date': next_date, 'temperature': temp})
        hist = pd.concat([hist, pd.DataFrame([{
            'date': next_date,
            'temperature': temp,
            'cloudiness': y_pred[1],
            'precipitation': y_pred[2],
            'solar_radiation': y_pred[3],
            'day_length_hours': feat['day_length_hours']
        }])], ignore_index=True)
    return hist, pd.DataFrame(forecasts)

model       = joblib.load(MODEL_PATH)
climatology = joblib.load(CLIMATOLOGY_PATH)
historical_df = (
    pd.read_csv(CSV_PATH, parse_dates=['date'])
      .sort_values('date')
      .reset_index(drop=True)
)
scale_factor  = joblib.load('scale_factor.pkl')

TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><title>Weather Forecast</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background: #f9f9f9; }
    h1 { color: #333; }
    form { margin-bottom: 20px; }
    input, button { padding: 8px; font-size: 1rem; }
    button { background: #007BFF; color: white; border: none; border-radius: 4px; }
    img { box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-radius: 8px; max-width:100%; }
    table { border-collapse: collapse; width: 100%; max-width: 800px; margin-top: 20px; background: white; }
    th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
    th { background-color: #f4f4f4; }
  </style>
</head>
<body>
  <h1>Historical &amp; Forecast</h1>
  <form method="get" action="/">
    <label>Start: <input type="date" name="start" value="{{ start }}"></label>
    <label>End:   <input type="date" name="end"   value="{{ end  }}"></label>
    <button type="submit">Update</button>
  </form>

  {% if plot %}
    <img src="data:image/png;base64,{{ plot }}" alt="Plot"/>
  {% endif %}

  {% if history_data %}
    <h2>Historical Data</h2>
    <table>
      <tr><th>Date</th><th>Temperature (°C)</th></tr>
      {% for row in history_data %}
        <tr>
          <td>{{ row.date.strftime('%Y-%m-%d') }}</td>
          <td>{{ '%.1f'|format(row.temperature) }}</td>
        </tr>
      {% endfor %}
    </table>
  {% endif %}

  {% if table_data %}
    <h2>Forecast</h2>
    <table>
      <tr><th>Date</th><th>Temperature (°C)</th></tr>
      {% for row in table_data %}
        <tr>
          <td>{{ row.date.strftime('%Y-%m-%d') }}</td>
          <td>{{ '%.1f'|format(row.temperature) }}</td>
        </tr>
      {% endfor %}
    </table>
  {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET'])
def index():
    start = request.args.get('start', '2025-01-01')
    end   = request.args.get('end',   '2025-12-31')
    start, end = pd.to_datetime(start), pd.to_datetime(end)

    last_hist_date = historical_df['date'].max()

    n_days = max(0, (end - last_hist_date).days)
    _, forecast_df = recursive_forecast(n_days, last_hist_date)

    history_df = historical_df[
        (historical_df['date'] >= start) &
        (historical_df['date'] <= min(end, last_hist_date))
    ]

    forecast_slice = forecast_df[
        (forecast_df['date'] > last_hist_date) &
        (forecast_df['date'] >= start) &
        (forecast_df['date'] <= end)
    ]

    fig, ax = plt.subplots(figsize=(12,5))
    if not history_df.empty:
        ax.plot(history_df['date'], history_df['temperature'],
                color='tab:blue', label='Historical')
    if not forecast_slice.empty:
        ax.plot(forecast_slice['date'], forecast_slice['temperature'],
                color='tab:red', linestyle='--', label='Forecast')
    ax.set_xlim(start, end)
    ax.set_xlabel('Date')
    ax.set_ylabel('Temperature (°C)')
    ax.legend()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('ascii')
    plt.close(fig)

    history_data = history_df[['date','temperature']].to_dict('records')
    table_data   = forecast_slice[['date','temperature']].to_dict('records')

    return render_template_string(
        TEMPLATE,
        plot=plot_data,
        history_data=history_data,
        table_data=table_data,
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d')
    )

if __name__ == '__main__':
    app.run(debug=True)