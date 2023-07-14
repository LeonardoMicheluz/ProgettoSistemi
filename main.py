#PROVA PRATICA DI SISTEMI DINAMICI
#MATTEO CAIOLA
#ALBERTO FACCHIN
#MICHELUZ LEONARDO

# Importazione delle librerie necessarie
import numpy as np
import matplotlib.pyplot as plt

import scipy
from sympy import *
from numpy.linalg import eig
from numba import njit, float64, int32, vectorize, guvectorize
from typing import Callable



# Funzione per la ODE
def f(x, y, a, b):
    dxdt = x * (3 - a * x - b * y)
    dydt = y * (2 - x - y)
    return np.array([dxdt, dydt])


# Funzione per la determinazione dei punti fissi e la loro stabilità
def puntiFissiTot():

    # Calcolo punti fissi
    def puntiFissi():
        x, y = symbols('x, y')
        eq1 = Eq(x * (3 - 2 * x - y), 0)
        eq2 = Eq(y * (2 - x - y), 0)
        sol = solve([eq1, eq2], [x, y])

        # Plot dei punti d'equilibrio
        for i, j in sol:
            plt.plot(i, j, marker="o", markersize=4, color="black")
        return sol

    # Calcolo la matrice Jacobiana
    def jacobian(variables):
        x, y = variables
        return [[3 - 4 * x - y, -x], [-y, 2 - x - 2 * y]]

    print(puntiFissi())  # Stampa i punti fissi

    # Calcolo le jacobiane nei punti fissi
    jacobians = [jacobian(point) for point in puntiFissi()]
    print(jacobians)

    # Calcolo gli autovalori e autovettori
    for jacobian in jacobians:
        eigenvalues, eigenvectors = eig(np.array(jacobian, dtype=float))

        if (eigenvalues[0] < 0 < eigenvalues[1]) or (eigenvalues[0] > 0 > eigenvalues[1]):
            print(eigenvectors[0], eigenvectors[1])


# Funzione per tracciare le isocline
def isocline():
    x1 = np.linspace(-1, 5)  # Linspace per generare punti
    y2 = np.linspace(-1, 5)
    y1 = -2*x1 + 3  # Prima eq.
    x2 = y2*0  # x=0
    y3 = -x1+2  # Seconda eq.
    y4 = x1*0  # y=0

    plt.ylim(-1, 5)
    plt.xlim(-1, 5)
    plt.plot(x1, y1, "r--", label="x-isocline")
    plt.plot(x2, y2, "r--")
    plt.plot(x1, y3, "g--", label="y-isocline")
    plt.plot(x1, y4, "g--")
    plt.plot(x1, y4, "g--")
    plt.legend(loc="upper right")

    # Tracciare l'orbita chiusa intorno al punto fisso
    #t = np.linspace(0, 2*np.pi, 65)
    #xc = 1 + 0.3 * np.cos(t)
    #yc = 1 + 0.3 * np.sin(t)
    #plt.plot(xc,yc, "k--")


# Funzione per plottare i campi vettoriali al variare di a e b
def plotdf(a_list, b_list, axs = None):
    xran = [-1, 5]
    yran = [-1, 5]
    grid = [21, 21]
    for i, ax in enumerate(axs.flat):
        a = a_list[i]
        b = b_list[i]

        x = np.linspace(xran[0], xran[1], grid[0])
        y = np.linspace(yran[0], yran[1], grid[1])

        X, Y = np.meshgrid(x, y)  # Crea la griglia
        DX, DY = f(X, Y, a, b)

        M = (np.hypot(DX, DY))  # Normalizza l'andamento della crescita
        M[M == 0] = 1.  # In caso di divisione per 0
        DX = DX / M  # Normalizza ogni freccia del campo
        DY = DY / M

        ax.quiver(X, Y, DX, DY, pivot='mid', color='orange')  # Plotta la griglia
        ax.set_title(f'a={a}, b={b}')


# Funzione per ottenere un singolo campo vettoriale
def plotdf_1(a, b):
    xran = [-1, 5]
    yran = [-1, 5]
    grid = [21, 21]

    x = np.linspace(xran[0], xran[1], grid[0])
    y = np.linspace(yran[0], yran[1], grid[1])

    X, Y = np.meshgrid(x, y)  # Crea la griglia
    DX, DY = f(X, Y, a, b)

    M = (np.hypot(DX, DY))  # Normalizza l'andamento della crescita
    M[M == 0] = 1.  # In caso di divisione per 0
    DX = DX / M  # Normalizza ogni freccia del campo
    DY = DY / M

    fig, ax = plt.subplots()
    ax.quiver(X, Y, DX, DY, pivot='mid', color='orange')  # Plotta la griglia
    ax.set_title(f'Campo vettoriale con a={a}, b={b} e isocline')
    fig.show()


# Piano di fase
def pianoFase(a, b):
    # Punti fissi
    fixed_points = np.array([[0., 0.], [0., 2.], [1.5, 0.], [1., 1.]])

    x = np.linspace(0, 3, 10)  # Generiamo punti
    y = np.linspace(0, 3, 10)
    X, Y = np.meshgrid(x, y)  # Generiamo la griglia

    # Funzione
    Xdot = X * (3 - a * X - b * Y)
    Ydot = Y * (2 - X - Y)

    fig, ax = plt.subplots()
    plt.title("Piano di fase")
    plt.ylabel("y")
    plt.xlabel("x")

    # Asse X e Y
    x1 = np.linspace(0, 3)  # Linspace per generare punti
    y2 = np.linspace(0, 3)
    x2 = y2 * 0  # x=0
    y4 = x1 * 0  # y=0
    plt.plot(x1, y4, color="k")
    plt.plot(x2, y2, color="k")

    ax.streamplot(X, Y, Xdot, Ydot, density = 0.8)
    ax.scatter(*fixed_points.T, color="r")
    fig.show()

def fg(x, a=2.0, b=1.0) -> np.ndarray[float64]:
    # Define the equations
    u = x[0] * (3 - a * x[0] - b * x[1])
    v = x[1] * (2 - x[0] - x[1])
    return np.array([u, v])

    # Funzione per rappresentare i manifold
def phase_diagram_trajectories(fg, dt: float64 = 0.1, final_time: float64 = 1, num: int = 20, start: int = -4, end: int = 4):
    # Generate all combinations of starting positions using nested for loops
    x_starts = np.linspace(start, end, num)
    y_starts = np.linspace(start, end, num)
    combinations = [(x, y) for x in x_starts for y in y_starts]

    # Initialize an array to store the trajectories
    trajectories = []

    # Solve the differential equation for each combination of x and y in a vectorized way
    x0_array = np.array(combinations)
    trajectories = vectorized_runge_kutta(fg, x0_array, dt, final_time)

    # Set up the plot
    fig, ax = plt.subplots()

    # Define the color map
    cmap = plt.get_cmap("viridis")

    # Plot the trajectories for each combination of x and y
    for i in range(len(combinations)):
        x_traj = trajectories[i][:, 0]
        y_traj = trajectories[i][:, 1]
        color = cmap(i / len(combinations))
        ax.plot(x_traj, y_traj, color=color, linewidth=0.5)

    # Add labels and title
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(start, end)
    ax.set_ylim(start, end)
    ax.set_title(f'Trajectories of the Differential Equation, dt = {dt}')

    # Set the background color to white
    fig.patch.set_facecolor("white")

    # Remove the top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add grid lines
    ax.grid(True, linestyle="--", color="gray", alpha=0.5)

    # Show the plot
    plt.show()


def rungekutta4(f, y0, t):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = 0.01
        k1 = f(y[i], t[i])
        k2 = f(y[i] + k1 * h / 2., t[i] + h / 2.)
        k3 = f(y[i] + k2 * h / 2., t[i] + h / 2.)
        k4 = f(y[i] + k3 * h, t[i] + h)
        y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y


def vectorized_runge_kutta(fg, x0_array, dt=0.001, final_time=1) -> np.ndarray:
    """Differential equations solver using Runge-Kutta method"""

    # Initialize the solution array
    num_iterations = int32(final_time / dt)
    num_conditions = x0_array.shape[0]
    x = np.empty((num_conditions, num_iterations, x0_array.shape[1]))
    x[:, 0] = x0_array

    for i in range(1, num_iterations):
        for j in range(num_conditions):
            k1 = fg(x[j, i - 1]) * dt
            k2 = fg(x[j, i - 1] + k1 / 2.0) * dt
            k3 = fg(x[j, i - 1] + k2 / 2.0) * dt
            k4 = fg(x[j, i - 1] + k3) * dt
            x[j, i] = x[j, i - 1] + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return x


#--------Main---------#

# Parametri
a = 2
b = 1

a_list = [1, 2, 1, 3]  # Parametri a e b per variare
b_list = [2, 1, 3, 1]


plt.rcParams['figure.dpi'] = 100  # Impostiamo graficamente i plot
plt.rcParams['font.size'] = 7
fig, axs = plt.subplots(2, 2)

plotdf(a_list, b_list, axs)  # Plotta 4 campi vettoriali

plotdf_1(a, b)

puntiFissiTot()  # Calcola i punti fissi

isocline()  # Traccia graficamente le isocline

pianoFase(a, b)  # Disegna il piano di fase

phase_diagram_trajectories(fg, dt=0.001, final_time=10, num=20, start=-0, end=3)
plt.show()
