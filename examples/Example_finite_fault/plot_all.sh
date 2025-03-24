# create folder
mkdir -p data img

# Verify the model
python verifyTerrain.py # Terrain
python plotXZ.py 0 Vs  # Vs, Slice Y
python plotYZ.py 0 Vs  # Vs, Slice X

# plot the wavefield snapshots
python plotXY.py 1000 Vx 1 # 1000th time step, Vx, Free surface
python plotXY.py 1000 Vx 0 # 1000th time step, Vx, Slice Z
python plotXZ.py 1000 Vx   # 1000th time step, Vx, Slice Y
python plotYZ.py 1000 Vx   # 1000th time step, Vx, Slice X

python plotXY.py 4600 Ux 1 # 4600th time step, Ux, Free surface

# plot waveforms recorded by stations
python plotWaveform.py

# plot PGVh and Intensity
python plotPGV.py PGVh # PGVh and corresponding Intensity, optional [PGVh, PGV, PGAh, PGA]
