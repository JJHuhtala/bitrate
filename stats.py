from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Load presaved trajectories. 
a = np.load("trajs.npy")
print(a[:,0,1])
flighttimes = []
print(np.max(a[:,0,0]))
aa = np.argmax(a[:,0,0])
for i in range(len(a)):
    for j in range(len(a[0,:,0])):
        if a[i,j,0] > 3.0:
            flighttimes.append(0.001*j)
            break
        if a[i,j,1 ] > 3.0:
            flighttimes.append(0.001*j)
            break

f = np.array(flighttimes)

print(f, np.min(f))
print(len(f))
print("Average: ", np.average(f), "Standard deviation: ", np.std(f))
#plt.hist(f,bins=10)
#plt.plot(a[8000,:,0])
#plt.show()

