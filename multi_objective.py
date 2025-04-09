import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
from scipy.optimize import minimize as opt
from matplotlib.colors import LogNorm
import matplotlib
from matplotlib import cm, ticker


# Problem statement:
#     - Using a method of your choice define the Pareto front of the multi-objective function f = [f1 f2]T.
#     - Plot at least 10 points on the Pareto front and then keep or add in 50 or more points that are dominated
#           (i.e. not on the Pareto front).
#     - Discuss the method you selected and its pros and cons in defining the Pareto-front in 200 to 400 words.

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

def Weighted_Sum(f,x0,the_bounds,n = 10):
    # get the bounds of the pareto front used in Weighted_sum
    # res1 = opt(f[0],x0,bounds=the_bounds)     
    # res2 = opt(f[1],x0,bounds=the_bounds)
    w = np.linspace(0,1,n)                      # uniform spacing in w, not uniform in f1 and f2
    def func_min(x,w):
        f1 = f[0](x)
        f2 = f[1](x)
        return w*f1 + (1-w)*f2

    # store all the points fora various weights
    solutions_x = np.array([])
    solutions_f = np.array([])
    for i in range(len(w)):
        res = opt(func_min, x0, args=(w[i],), bounds=the_bounds)
        if(i == 0):
            solutions_x = res.x
            solutions_f = np.array([f[0](res.x),f[1](res.x)])
        else:
            solutions_x = np.vstack([solutions_x,res.x])
            solutions_f = np.vstack([solutions_f,[f[0](res.x),f[1](res.x)]])
    
    return solutions_x, solutions_f
    

def Normal_Boundary_Intersection(f,x0,the_bounds):
    # get the bounds of the pareto front used in Normal_boundary_intersection
    res1 = opt(f[0],x0,bounds=the_bounds)     
    res2 = opt(f[1],x0,bounds=the_bounds)

    f_star_utp = np.array([res1.x[0],res2.x[1]])
    P = np.array([])
    for i in range(np.size(f)):
        res = opt(f[i],x0,bounds=the_bounds)
        f_star = res.x
        if(i == 0):
            P = f_star-f_star_utp
        else:
            P = np.vstack([P, f_star - f_star_utp])
    n_tild = np.zeros(np.size(x0))
    for i in range(np.size(x0)):
        n_tild[i] = -np.max(np.abs(P[i]))

    e = np.ones(np.size(x0))
    alpha = np.ones(np.size(x0))
    def func_min(alpha):
        step = np.matmul(P,e) + f_star_utp + alpha*n_tild
        step_norm = np.linalg.norm(step)
        return 1/step_norm

    res_int = opt(func_min, alpha)
    alpha_step = res_int.x


f1 = sphere_function
f2 = Goldstein_price_function

the_bounds = ((-2,2),(-3,1))

x0 = [0,-1]
res = opt(f1,x0)
res2 = opt(f2,x0,bounds = the_bounds)
Normal_Boundary_Intersection([f1,f2],x0,the_bounds)

n_pareto_points = 50000
x_stars,f_stars = Weighted_Sum([f1,f2],x0,the_bounds,n_pareto_points = 50000)

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
Z3 = np.zeros_like(X2)  # stores the values of f1 for the same grid
for i in range(len(x)-1):
    for j in range(len(x)-1):
        Z2[i,j] = f2([x[i],y[j]])
        Z3[i,j] = f1([x[i],y[j]])

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
# plt.show()

fig,ax = plt.subplots()
# cs = ax.contourf(X2, Y2, Z3, levels=nlevels, cmap='viridis')
cs1 = ax.contour(X2, Y2, Z3, levels=nlevels, linewidths=2,cmap='magma')
# plt.scatter(res.x[0], res.x[1], color='red', marker='*', s=100, label='Optimum')
cs2 = ax.contourf(X2, Y2, Z2, levels=nlevels, locator=ticker.LogLocator(), cmap="viridis", alpha=1)
# ax.contour(X2, Y2, Z2, levels=300, colors='black', linewidths=0.5,alpha=0.5)
plt.scatter(res.x[0], res.x[1], color='red', marker='*', s=50, label='f1*')
plt.scatter(res2.x[0], res2.x[1], color='blue', marker='*', s=50, label='f2*')
plt.scatter(x_stars[:,0],x_stars[:,1], color='pink', marker='o', s=10, label='Pareto Front')
plt.plot(x_stars[:,0],x_stars[:,1], color='pink')
c1 = plt.colorbar(cs1)
c1.set_label('f1')
c2 = plt.colorbar(cs2)
c2.set_label('f2')
plt.legend()
plt.show()