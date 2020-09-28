'''Plots and example weighting function'''
import numpy as np
import matplotlib.pyplot as plt
import misc

pressure = np.exp(np.linspace(np.log(1000), np.log(5), 300))

H = 8000
q = 0.01
g = 9.81
k = 7

Z = -H*np.log(pressure/1000)/1000

tau = k*q/g * pressure
T= np.exp(-tau)

plt.plot(T, Z, label='Transmissivity')
plt.plot(pressure/1000, Z, label='Pressure (atm)')
dTdZ = np.diff(T)/np.diff(Z)
plt.plot(dTdZ/np.max(dTdZ), misc.stats.lin_av(Z), label='Weighting fn')
plt.axhline(15.7, lw=0.5, c='k')

#plt.plot(0.9*(T*pressure/1000)/np.max(T*pressure/1000), Z)

plt.ylabel('Pressure height (km)')
plt.xlim(0, 1.05)
plt.ylim(0, 40)
plt.legend()

fig = plt.gcf()
fig.set_size_inches(4,4)
fig.savefig('output/weighting_fn_co2.pdf', bbox_inches='tight')


