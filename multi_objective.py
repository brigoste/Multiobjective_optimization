import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

def sphere_function(x):
    return np.sum(x**2)
def Rosenbrock_function(x):
    sum = 0
    for i in range(len(x)-1):
        sum = sum +  100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2
    return sum

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = sphere_function([X, Y])
Z2 = Rosenbrock_function([X, Y])

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar()
plt.title('Sphere Function')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.subplot(1, 2, 2)
plt.contourf(X, Y, Z2, levels=50, cmap='viridis')
plt.colorbar()
plt.title('Rosenbrock Function')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.tight_layout()
plt.show()
