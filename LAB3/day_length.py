import pandas as pd
import numpy as np
from astropy.time import Time
from astropy.coordinates import get_sun, EarthLocation
import astropy.units as u

LATITUDE = 47.54694
LONGITUDE = 7.56892

daily_in = 'daily_weather.csv'
daily_out = 'daily_weather_with_daylength.csv'

df = pd.read_csv(daily_in, parse_dates=['date'])

location = EarthLocation(lat=LATITUDE * u.deg, lon=LONGITUDE * u.deg)

def compute_day_length(date):
    t = Time(f"{date.date()} 12:00:00")
    sun = get_sun(t)
    dec = sun.dec.rad
    phi = location.lat.rad
    cos_omega0 = -np.tan(phi) * np.tan(dec)
    cos_omega0 = np.clip(cos_omega0, -1.0, 1.0)
    omega0 = np.arccos(cos_omega0)
    return 2 * omega0 * 24 / (2 * np.pi)

df['day_length_hours'] = df['date'].apply(compute_day_length)
df.to_csv(daily_out, index=False)
print(f"Saved augmented CSV to {daily_out}")