import struct
f = open('./source/Chenkejie/Chenkejie_source106.bin','rb')
npts = struct.unpack('i', f.read(4))[0]
nt   = struct.unpack('i', f.read(4))[0]
dt   = struct.unpack('f', f.read(4))[0]
print('npts=%d  nt=%d  dt=%g s  duration=%g s' % (npts, nt, dt, nt*dt))
print('震源结束步 =', int((nt-1)*dt/0.013))

