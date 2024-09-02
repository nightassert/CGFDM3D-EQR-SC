# plot snapshot on free surface
python plotXY.py 5000 Uz 1 SCFDM
python plotXY.py 2900 Uz 1 CGFDM

# plot waveform on stations
python plotMultiSeismos.py Vx Ux CGFDM
python plotMultiSeismos.py Vy Uy CGFDM
python plotMultiSeismos.py Vz Uz CGFDM

python plotMultiSeismos.py Vx Ux SCFDM
python plotMultiSeismos.py Vy Uy SCFDM
python plotMultiSeismos.py Vz Uz SCFDM
python plot_waveform.py