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
targets = ['HEDEF_1', 'HEDEF_2', 'HEDEF_3']
for t in targets:
    t_data = sensor[sensor['callsign_true'] == t]
    real = (t_data['is_clutter'] == False).sum()
    clutter = (t_data['is_clutter'] == True).sum()
    if len(t_data) > 0:
        tmin, tmax = t_data["time"].min(), t_data["time"].max()
        print(f'{t:20s}: {real:4d} real, {clutter:4d} clutter (times: {tmin:.1f} - {tmax:.1f} s)')

# Speed analysis for HEDEF_3
h3_real = sensor[(sensor['callsign_true'] == 'HEDEF_3') & (sensor['is_clutter'] == False)]
print()
print('HEDEF_3 speed analysis (REAL measurements only):')
if len(h3_real) > 0:
    h3_real['speed'] = (h3_real['vx']**2 + h3_real['vy']**2)**0.5
    print(f'  Speed: min={h3_real["speed"].min():.1f}, max={h3_real["speed"].max():.1f}, mean={h3_real["speed"].mean():.1f} m/s')
    print(f'  # measurements: {len(h3_real)}')
    print(f'  # radars: {h3_real["sensor"].nunique()}')
    
    # Check per radar
    for radar in sorted(h3_real['sensor'].unique()):
        radar_data = h3_real[h3_real['sensor'] == radar]
        print(f'    {radar}: {len(radar_data)} measurements')
else:
    print(f'  NO REAL MEASUREMENTS FOR HEDEF_3!')
