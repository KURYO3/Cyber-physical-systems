import pandas as pd
import numpy as np
import joblib
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

df = pd.read_csv('daily_weather_with_daylength.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

climatology = df.groupby(df['date'].dt.dayofyear)['temperature'].mean()

df['temp_residual'] = df.apply(
    lambda row: row['temperature'] - climatology.loc[row['date'].dayofyear],
    axis=1
)

df['dayofyear'] = df['date'].dt.dayofyear
df['month']     = df['date'].dt.month
df['weekday']   = df['date'].dt.weekday
df['sin_doy']   = np.sin(2 * np.pi * df['dayofyear'] / 365)
df['cos_doy']   = np.cos(2 * np.pi * df['dayofyear'] / 365)
df['temp_roll3']  = df['temp_residual'].shift(1).rolling(3, min_periods=1).mean()
df['temp_roll7']  = df['temp_residual'].shift(1).rolling(7, min_periods=1).mean()
df['temp_trend7'] = df['temp_residual'] - df['temp_roll7']
bins   = [0, 8, 12, 16, 24]
labels = ['deep_winter','winter_spring','spring_summer','long_summer']
df['season_dl']  = pd.cut(df['day_length_hours'], bins=bins, labels=labels)
df = pd.get_dummies(df, columns=['season_dl'])
df['dl_x_sin'] = df['day_length_hours'] * df['sin_doy']
df['dl_x_cos'] = df['day_length_hours'] * df['cos_doy']
df['temp_lag_14']  = df['temp_residual'].shift(14)
df['temp_roll14']  = df['temp_residual'].shift(1).rolling(14, min_periods=1).mean()
df['temp_roll28']  = df['temp_residual'].shift(1).rolling(28, min_periods=1).mean()
for col in ['solar_radiation','precipitation','cloudiness']:
    df[f'{col}_roll7'] = df[col].shift(1).rolling(7, min_periods=1).mean()
for lag in range(1, 8):
    for col in ['temp_residual','solar_radiation','precipitation','cloudiness','day_length_hours']:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)
df = df.dropna().reset_index(drop=True)

cutoff = pd.Timestamp('2024-01-01')
train = df[df['date'] < cutoff]
test  = df[df['date'] >= cutoff]

y_cols = ['temp_residual','cloudiness','precipitation','solar_radiation']
X_cols = [c for c in df.columns if c not in ['date','temperature'] + y_cols]

X_train, y_train = train[X_cols], train[y_cols]
X_test,  y_test  = test[X_cols],  test[y_cols]

threshold = 1.0
abs_resid = np.abs(y_train['temp_residual'])
weights = np.where(abs_resid > threshold, 1.0 + abs_resid, 1.0)

base = lgb.LGBMRegressor(
    num_leaves=50, max_depth=15, learning_rate=0.01,
    n_estimators=200, feature_fraction=1.0,
    bagging_fraction=0.8, bagging_freq=5,
    n_jobs=-1, random_state=42
)
model = MultiOutputRegressor(base, n_jobs=-1)
model.fit(X_train, y_train, **{'sample_weight': weights})

preds = model.predict(X_test)
for i, col in enumerate(y_cols):
    mae = mean_absolute_error(y_test[col], preds[:,i])
    print(f"{col} MAE: {mae:.3f}")

y_true_resid = y_train['temp_residual']
y_pred_resid = model.predict(X_train)[:, 0]
scale_factor = y_true_resid.std() / (y_pred_resid.std() or 1.0)
print(f"Residual scaling factor: {scale_factor:.3f}")

joblib.dump(model, 'weather_multioutput.pkl')
joblib.dump(climatology, 'climatology.pkl')
joblib.dump(scale_factor, 'scale_factor.pkl')
print("Saved model, climatology and scale_factor.")
