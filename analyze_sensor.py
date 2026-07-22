#!/usr/bin/env python3
import pandas as pd

# Sensor data analizi
sensor = pd.read_csv('radar_sensor_tracks_gercekci.csv')
print('SENSOR DATA ANALYSIS')
print('='*60)
print(f'Total rows: {len(sensor)}')
print(f'Time range: {sensor["time"].min():.2f} - {sensor["time"].max():.2f} s')
print()

# is_clutter breakdown
print('is_clutter breakdown:')
print(f'  is_clutter=False (real): {(sensor["is_clutter"] == False).sum()}')
print(f'  is_clutter=True (clutter): {(sensor["is_clutter"] == True).sum()}')
print()

# Per-target
targets = sorted(sensor['callsign_true'].dropna().unique())
for t in targets:
    t_data = sensor[sensor['callsign_true'] == t]
    real = (t_data['is_clutter'] == False).sum()
    clutter = (t_data['is_clutter'] == True).sum()
    if len(t_data) > 0:
        tmin, tmax = t_data["time"].min(), t_data["time"].max()
        print(f'{t:20s}: {real:4d} real, {clutter:4d} clutter (times: {tmin:.1f} - {tmax:.1f} s)')

print()
print('Per-target speed analysis (REAL measurements only):')
for target in targets:
    t_real = sensor[(sensor['callsign_true'] == target) & (sensor['is_clutter'] == False)].copy()
    print(f'  {target}:')
    if len(t_real) > 0:
        t_real['speed'] = (t_real['vx']**2 + t_real['vy']**2)**0.5
        print(f'    Speed: min={t_real["speed"].min():.1f}, max={t_real["speed"].max():.1f}, mean={t_real["speed"].mean():.1f} m/s')
        print(f'    # measurements: {len(t_real)}')
        print(f'    # radars: {t_real["sensor"].nunique()}')
        for radar in sorted(t_real['sensor'].unique()):
            radar_data = t_real[t_real['sensor'] == radar]
            print(f'      {radar}: {len(radar_data)} measurements')
    else:
        print(f'    NO REAL MEASUREMENTS FOR {target}!')
