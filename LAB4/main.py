import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

SIGMA = 10.0
RHO   = 28.0
BETA  = 8.0 / 3.0

def lorenz(state, t, sigma=SIGMA, rho=RHO, beta=BETA):
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

def simulate(initial_state, t):
    return odeint(lorenz, initial_state, t)

def plot_3d(sol1, sol2, t):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol1[:,0], sol1[:,1], sol1[:,2],
            label='Траєкторія 1', lw=0.5, color='blue')
    ax.plot(sol2[:,0], sol2[:,1], sol2[:,2],
            label='Траєкторія 2', lw=0.5, color='orange')
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Дві траєкторії атрактора Лоренца")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_diff(sol1, sol2, t):
    dist = np.linalg.norm(sol1 - sol2, axis=1)
    plt.figure(figsize=(10,5))
    plt.plot(t, dist, label="Різниця між траєкторіями")
    plt.yscale('log')
    plt.xlabel("Час"); plt.ylabel("Різниця")
    plt.title("Розбіжність траєкторій")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    t = np.linspace(0, 30, 10000)

    state0 = [1.0, 1.0, 1.0]
    state1 = [1.0001, 1.0, 1.0]

    sol1 = simulate(state0, t)
    sol2 = simulate(state1, t)

    plot_3d(sol1, sol2, t)
    plot_diff(sol1, sol2, t)

if __name__ == "__main__":
    main()
