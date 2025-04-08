import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
from scipy.optimize import minimize as opt
from matplotlib.colors import LogNorm
import matplotlib
from matplotlib import cm, ticker

def sphere_function(x):
    sum = 0
    for i in x:
        sum = sum + i**2
    return sum
def Goldstein_price_function(x):
    y = x[1]
    x = x[0]
    t1 = 1 + (x+y+1)**2 * (19 - 14*x + 3*(x**2) - 14*y + 6*x*y + 3*(y**2))
    t2 = 30 + ((2*x - 3*y)**2)*(18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
    return t1 * t2

f1 = sphere_function
f2 = Goldstein_price_function

the_bounds = ((-2,2),(-3,1))

x0 = [0,-1]
res = opt(f1,x0)
res2 = opt(f2,x0,bounds = the_bounds)

print(res)
print(res2)

x = np.linspace(-4.5, 4.5, 100)
y = x
Y,X = np.meshgrid(x, y)
Z1 = np.zeros_like(X)
for i in range(len(x)-1):
    for j in range(len(x)-1):
        Z1[i,j] = f1([x[i],y[j]])

y = np.linspace(-2,1,1000)
x = np.linspace(-2,2,1000)
Y2,X2 = np.meshgrid(y, x)
Z2 = np.zeros_like(X2)
for i in range(len(x)-1):
    for j in range(len(x)-1):
        Z2[i,j] = f2([x[i],y[j]])

nlevels = 500
fig,ax = plt.subplots()
cs = ax.contourf(X2, Y2, Z2, levels=nlevels, locator=ticker.LogLocator(), cmap='viridis')
ax.contour(X2, Y2, Z2, levels=300, colors='black', linewidths=0.5,alpha=0.5)
plt.scatter(res2.x[0], res2.x[1], color='red', marker='*', s=100, label='Optimum')
fig.colorbar(cs)
plt.title(f2.__name__)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

nlevels = 20

fig,ax = plt.subplots()
cs = ax.contourf(X, Y, Z1, levels=nlevels, cmap='viridis')
ax.contour(X, Y, Z1, levels=nlevels, colors='black', linewidths=0.5)
plt.scatter(res.x[0], res.x[1], color='red', marker='*', s=100, label='Optimum')
fig.colorbar(cs)
plt.title(f1.__name__)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

