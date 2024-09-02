# plot snapshot on free surface
python plotXZ.py 1700 Vx CGFDM
python plotXZ.py 1700 Vy CGFDM
python plotXZ.py 1700 Vz CGFDM

python plotXZ.py 1700 Vx SCFDM
python plotXZ.py 1700 Vy SCFDM
python plotXZ.py 1700 Vz SCFDM

# plot waveform on stations
python plotMultiSeismos.py Vx Ux CGFDM
python plotMultiSeismos.py Vy Uy CGFDM
python plotMultiSeismos.py Vz Uz CGFDM

python plotMultiSeismos.py Vx Ux SCFDM
python plotMultiSeismos.py Vy Uy SCFDM
python plotMultiSeismos.py Vz Uz SCFDM
python plot_waveform.py