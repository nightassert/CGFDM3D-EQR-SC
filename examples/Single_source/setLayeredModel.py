#!/usr/bin/env python

"""
Created on 2022-10-08
17:21
@author: Wenqiang Wang 
"""

import numpy as np
import matplotlib.pyplot as plt
import struct
import sys
import os
import json


def vs2vp(Vs):
    Vp = (0.9409 + 2.0947 * Vs - 0.8206 * (Vs ** 2) +
          0.2683 * (Vs ** 3) - 0.0251 * (Vs ** 4))
    return Vp


def vp2rho(Vp):
    rho = (1.6612 * Vp - 0.4721 * (Vp ** 2) + 0.0671 *
           (Vp ** 3) - 0.0043 * (Vp ** 4) + 0.000106 * (Vp ** 5))
    return rho


def loadFromFile(fileName):
    data = np.loadtxt(fileName)
    return data


fileName = "Layered_Model.txt"

Trans = 1

# Dep = np.zeros( [Layers] )
# Vs  = np.zeros( [Layers] )
# Vp  = np.zeros( [Layers] )
# Rho = np.zeros( [Layers] )
data = loadFromFile(fileName).T

Layers = np.shape(data[0, :])[0]

Dep = np.zeros([Layers])
Vs = np.zeros([Layers])
Vp = np.zeros([Layers])
Rho = np.zeros([Layers])

Dep = data[0, :]
Vs = data[2, :]

print("Dep:")
print(Dep)

if Trans == 1:
    Vp = data[1, :]
    Rho = data[3, :]
else:
    Vs[0] = Vs[1]
    for l in range(Layers):
        vs = Vs[l]
        vp = vs2vp(vs)
        rho = vp2rho(vp)
        Vp[l] = vp
        Rho[l] = rho

# Dep /= 1000
# Vs /= 1000
# Vp /= 1000
# Rho /= 1000

jsonsFile = open("./params.json")
params = json.load(jsonsFile)


LayeredModel = params["LayeredModel"]
LayeredFileName = params["LayeredFileName"]

LayeredFile = open(LayeredFileName, "wb")


value = struct.pack("i", Layers)
LayeredFile.write(value)

for i in range(Layers):
    value = struct.pack("f", Dep[i])
    LayeredFile.write(value)
    value = struct.pack("f", Vs[i])
    LayeredFile.write(value)
    value = struct.pack("f", Vp[i])
    LayeredFile.write(value)
    value = struct.pack("f", Rho[i])
    LayeredFile.write(value)

LayeredFile.close()
