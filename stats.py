from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
a = np.load("testtraj.npy")
print(len(a[0,:,0]))
flighttimes = []
print(np.max(a[:,0,0]))

for i in range(len(a)):
    for j in range(len(a[0,:,0])):
        if a[i,j,0] > 3.0:
            flighttimes.append(0.001*j)
            break

for i in range(len(a)):
    for j in range(len(a[0,:,0])):
        if a[i,j,1] > 3.0:
            flighttimes.append(0.001*j)
            break

f = np.array(flighttimes)
"""print(a[-1,:,0])
plt.plot(a[-1,:,0])
plt.show()
a,b,c = stats.binned_statistic(f,f,"count", bins=20)
print(len(f))
print(f)
plt.plot(a)
plt.show()"""
print(f, np.min(f))
plt.hist(f,bins=20)
plt.show()