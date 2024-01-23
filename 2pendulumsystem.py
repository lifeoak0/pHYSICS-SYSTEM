import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')  # 或者尝试其他后端，如 'Qt5Agg', 'GTK3Agg', 等。
import matplotlib.pyplot as plt


# Constants
g = 9.81  # acceleration due to gravity
L1 = 1.0  # length of pendulum 1
L2 = 1.0  # length of pendulum 2
m1 = 1.0  # mass of pendulum 1
m2 = 1.0  # mass of pendulum 2

# Initial conditions
theta1 = np.pi / 2
theta2 = np.pi / 2
omega1 = 0
omega2 = 0

# Time array
t_start = 0
t_end = 20
dt = 0.05
t = np.arange(t_start, t_end + dt, dt)


# Equations of motion
def dSdt(t, S):
    theta1, z1, theta2, z2 = S
    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)

    theta1_dot = z1
    z1_dot = (m2 * g * np.sin(theta2) * c - m2 * s * (L1 * z1 ** 2 * c + L2 * z2 ** 2) -
              (m1 + m2) * g * np.sin(theta1)) / L1 / (m1 + m2 * s ** 2)
    theta2_dot = z2
    z2_dot = ((m1 + m2) * (L1 * z1 ** 2 * s - g * np.sin(theta2) + g * np.sin(theta1) * c) +
              m2 * L2 * z2 ** 2 * s * c) / L2 / (m1 + m2 * s ** 2)

    return [theta1_dot, z1_dot, theta2_dot, z2_dot]


# Initial state
S0 = [theta1, omega1, theta2, omega2]

# Solve ODE
sol = solve_ivp(dSdt, [t_start, t_end], S0, t_eval=t, method='RK45')

# Extract the solutions
theta1, omega1, theta2, omega2 = sol.y

# Convert to Cartesian coordinates
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = L2 * np.sin(theta2) + x1
y2 = -L2 * np.cos(theta2) + y1

# Create animation
fig, ax = plt.subplots()
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
line, = ax.plot([], [], 'o-', lw=2)


def init():
    line.set_data([], [])
    return line,


def animate(i):
    line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
    return line,


ani = FuncAnimation(fig, animate, frames=lenwosh(t), init_func=init, blit=True)
plt.show()
