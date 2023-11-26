import numpy as np
import matplotlib.pyplot as plt
a = np.loadtxt('danet.txt')

x = a[0:20,[1,2,3]]
y = a[0:20,[0]]

#c = np.hstack([x, np.ones(y.shape)])
c = x
v = np.linalg.pinv(c) @ y

print(v)


plt.plot(y, 'r-')
plt.plot(v[0]*x[:,0] + v[1]*x[:,1] + v[2]*x[:,2],'g-')
plt.plot(x[:,0], 'b-')
plt.show()


