import numpy as np
from scipy import sparse
import sympy
import matplotlib.pyplot as plt
import matplotlib
import progressbar
from cycler import cycler
import time
from sklearn.metrics import mean_squared_error as MSE


N = 1000
dx = 2*np.pi/N
dt = 2.5e-4
nu=0.05


x = np.arange(0, 2*np.pi, dx)
t = np.arange(0, 0.5+dt, dt)

#-------------------------------------------------------------------------------
def analytic_method():
    X, T = sympy.symbols('X T')
    phi = sympy.exp(-(X - 4 * T)**2 / (4 * nu * (T + 1))) + sympy.exp(-(X - 4 * T - 2 * np.pi)**2 / (4 * nu * (T + 1)))
    dphidx = phi.diff(X)

    u_analytic_ = -2 * nu / phi * dphidx + 4
    u_analytic  = sympy.utilities.lambdify((X, T), u_analytic_)

    return u_analytic


analytic_solution = analytic_method()


def u_analytic(t, x):
    return analytic_solution(x, t)


def FDM(ui):
    ''' FDM with first order (Euler) approximation
    '''
    u0 = ui[:-2]
    u1 = ui[1:-1]
    u2 = ui[2:]
    return u1 + dt*( -u1*(u1-u0)/dx + nu*(u2-2*u1+u0)/dx**2)


def set_colors(start_color="blue"):
    cvals  = [0., 1]
    colors = [start_color, "violet"]
    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    ax = plt.axes()
    ax.set_prop_cycle(None)

    colors = [cmap(i) for i in np.linspace(0, 1, 8)[:6][::-1]]
    ax.set_prop_cycle(cycler('color', colors))
    return


def plot(x, u, dt, label, linestyle='solid', alpha=1, time_interval=0.1):
    step = int(time_interval/dt)
    u = u[::step]
    for un in u[:-1]:
        plt.plot(x, un, linestyle=linestyle, alpha=alpha)
    plt.plot(x, un, linestyle=linestyle, label=label, alpha=alpha)
    return
#-------------------------------------------------------------------------------


def main():
    u0 = u_analytic(0, x)

    u_prec = np.empty((t.shape[0], x.shape[0]))
    u_prec[0] = u_analytic(0, x)
    for i in range(1, t.shape[0]):
        u_prec[i] = u_analytic(t[i], x)

    u_fdm = np.empty((t.shape[0], x.shape[0]))
    u_fdm[0] = u_analytic(0, x)
    u_fdm[:, 0] = u_analytic(t, x[0])
    u_fdm[:, -1] = u_analytic(t, x[-1])

    start = time.time()
    for i in range(1, t.shape[0]):
        u_fdm[i, 1:-1] = FDM(u_fdm[i-1])
    end = time.time()
    print('elapsed time (FDM):', end-start)
    print('MSE (FDM):', MSE(u_fdm[-1], u_prec[-1]))


    set_colors()
    plot(x, u_fdm, dt, 'FDM (dt=2.5e-4)')
    plot(x, u_prec, dt, 'Analytic', linestyle='dashed', alpha=0.5)


    plt.grid()
    plt.ylim([0, 10])
    plt.xlim([0, 2*np.pi])
    plt.xlabel('x')
    plt.ylabel('u(t,x)')
    plt.legend()
    plt.show()
    return 0


if __name__ == "__main__":
    main()


