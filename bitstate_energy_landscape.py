# bitstate_energy_landscape.py
# Generates a 3D energy landscape of bit-states and a Navigator Cortex trajectory
# Saves the output as a PNG file

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# Define the energy landscape
# -----------------------------
def energy(x, y):
    """Energy function: high peaks = noisy, low valleys = meaningful"""
    return np.sin(3*x)*np.cos(3*y) + (x**2 + y**2)/5

# Generate grid
x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)
Z = energy(X, Y)

# -----------------------------
# Define Navigator Cortex path
# -----------------------------
# Example path from high energy to low energy
path_x = np.linspace(-1.5, 0.5, 50)
path_y = 0.5*np.sin(2*path_x)
path_z = energy(path_x, path_y)

# -----------------------------
# Plotting
# -----------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')

# Trajectory plot
ax.plot(path_x, path_y, path_z, color='red', linewidth=3, label='Navigator Cortex', marker='o')

# Labels and title
ax.set_xlabel('Bit-space X')
ax.set_ylabel('Bit-space Y')
ax.set_zlabel('Energy')
ax.set_title('Bit-State Universe Energy Landscape with Navigator Cortex')
ax.legend()

# Colorbar
fig.colorbar(surf, shrink=0.5, aspect=10, label='Energy Level')

# Save figure as PNG
plt.savefig("bitstate_energy_landscape.png", dpi=300)
plt.show()