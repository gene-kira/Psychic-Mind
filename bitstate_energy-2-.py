#!/usr/bin/env python3
# Borg‑OS Visualization Generator (Full Version)
# Generates PNGs, GIF, and interactive HTML in the current directory.

import importlib
import subprocess
import sys

# ---------------------------------------------------------
# AUTOLOADER: installs missing libraries automatically
# ---------------------------------------------------------
required = [
    "numpy",
    "matplotlib",
    "networkx",
    "scipy",
    "Pillow",
    "imageio",
    "plotly"
]

def ensure_libs():
    for lib in required:
        try:
            importlib.import_module(lib)
        except ImportError:
            print(f"[Autoloader] Installing missing library: {lib}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

ensure_libs()

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
from scipy.spatial.distance import cdist
from PIL import Image
import imageio
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. 3D ENERGY SURFACE
# ---------------------------------------------------------
def generate_energy_surface():
    X = np.linspace(-3, 3, 200)
    Y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(X, Y)
    Z = np.sin(X) * np.cos(Y) + 0.2 * np.sin(3 * X) + 0.2 * np.cos(3 * Y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.set_title("3D Energy Surface")
    ax.set_xlabel("Bit‑space X")
    ax.set_ylabel("Bit‑space Y")
    ax.set_zlabel("Energy")
    plt.savefig("energy_surface_3d.png", dpi=300)
    plt.close()
    print("PNG generated: energy_surface_3d.png")

# ---------------------------------------------------------
# 2. NAVIGATOR CORTEX MULTI-PATH OVERLAY
# ---------------------------------------------------------
def generate_navigator_paths():
    np.random.seed(0)
    N = 30
    pts = np.random.rand(N, 2)
    energy = np.random.rand(N)

    G = nx.DiGraph()
    for i in range(N):
        G.add_node(i, pos=pts[i], energy=energy[i])

    dist = cdist(pts, pts)
    for i in range(N):
        nearest = np.argsort(dist[i])[1:4]
        best = min(nearest, key=lambda j: energy[j])
        G.add_edge(i, best)

    pos = {i: pts[i] for i in range(N)}
    colors = energy
    sizes = (1 - energy) * 800 + 200

    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap="viridis", node_size=sizes)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", alpha=0.6)
    plt.title("Navigator Cortex Multi‑Path Overlay")
    plt.axis("off")
    plt.savefig("navigator_multipath.png", dpi=300)
    plt.close()
    print("PNG generated: navigator_multipath.png")

# ---------------------------------------------------------
# 3. SWARM-MESH DISTRIBUTED SEARCH
# ---------------------------------------------------------
def generate_swarm_mesh():
    np.random.seed(1)
    N = 40
    pts = np.random.rand(N, 2)
    G = nx.random_geometric_graph(N, radius=0.25)
    pos = {i: pts[i] for i in range(N)}

    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, node_size=200, node_color="cyan", edge_color="gray")
    plt.title("Swarm‑Mesh Distributed Search Diagram")
    plt.axis("off")
    plt.savefig("swarm_mesh.png", dpi=300)
    plt.close()
    print("PNG generated: swarm_mesh.png")

# ---------------------------------------------------------
# 4. MEMORY BASIN LIBRARY
# ---------------------------------------------------------
def generate_memory_basins():
    np.random.seed(2)
    N = 25
    pts = np.random.rand(N, 2)
    energy = np.random.rand(N)
    basin_centers = pts[np.argsort(energy)[:3]]

    plt.figure(figsize=(10, 7))
    plt.scatter(pts[:, 0], pts[:, 1], c=energy, cmap="plasma", s=300)
    plt.scatter(basin_centers[:, 0], basin_centers[:, 1], c="white", s=600, edgecolors="black")
    plt.title("Memory Basin Library")
    plt.colorbar(label="Energy")
    plt.savefig("memory_basins.png", dpi=300)
    plt.close()
    print("PNG generated: memory_basins.png")

# ---------------------------------------------------------
# 5. BIT-GEOMETRY EMBEDDING
# ---------------------------------------------------------
def generate_bit_geometry_embedding():
    np.random.seed(3)
    N = 50
    bits = np.random.randint(0, 2, (N, 16))
    pts = np.random.rand(N, 2)
    dist = cdist(bits, bits, metric="hamming")

    plt.figure(figsize=(10, 7))
    plt.scatter(pts[:, 0], pts[:, 1], c=dist.mean(axis=1), cmap="cool", s=200)
    plt.title("Bit‑Geometry Embedding")
    plt.colorbar(label="Mean Hamming Distance")
    plt.savefig("bit_geometry_embedding.png", dpi=300)
    plt.close()
    print("PNG generated: bit_geometry_embedding.png")

# ---------------------------------------------------------
# 6. COLLAPSE DYNAMICS (FRAME-BY-FRAME PNGs)
# ---------------------------------------------------------
def generate_collapse_dynamics():
    np.random.seed(4)
    N = 20
    pts = np.random.rand(N, 2)
    energy = np.random.rand(N)
    target = np.argmin(energy)
    target_pt = pts[target]

    for frame in range(20):
        alpha = frame / 19
        moved = pts * (1 - alpha) + target_pt * alpha
        plt.figure(figsize=(8, 6))
        plt.scatter(moved[:, 0], moved[:, 1], c=energy, cmap="viridis", s=300)
        plt.scatter(target_pt[0], target_pt[1], c="red", s=600, edgecolors="black")
        plt.title(f"Collapse Dynamics Frame {frame+1}")
        plt.savefig(f"collapse_dynamics_{frame:03d}.png", dpi=200)
        plt.close()
    print("Frames generated: collapse_dynamics_000.png to collapse_dynamics_019.png")

# ---------------------------------------------------------
# ENHANCEMENTS
# ---------------------------------------------------------
def generate_collapse_gif():
    filenames = [f"collapse_dynamics_{i:03d}.png" for i in range(20)]
    images = [imageio.imread(fname) for fname in filenames]
    imageio.mimsave("collapse_dynamics.gif", images, duration=0.2)
    print("GIF generated: collapse_dynamics.gif")

def generate_energy_colored_paths():
    np.random.seed(0)
    N = 30
    pts = np.random.rand(N, 2)
    energy = np.random.rand(N)
    G = nx.DiGraph()
    for i in range(N):
        G.add_node(i, pos=pts[i], energy=energy[i])
    dist = cdist(pts, pts)
    for i in range(N):
        nearest = np.argsort(dist[i])[1:4]
        best = min(nearest, key=lambda j: energy[j])
        G.add_edge(i, best)
    pos = {i: pts[i] for i in range(N)}
    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_color=energy, cmap="plasma", node_size=300)
    nx.draw_networkx_edges(G, pos, arrows=True, alpha=0.6)
    plt.title("Navigator Cortex Nodes Colored by Energy")
    plt.axis("off")
    plt.savefig("navigator_energy_colored.png", dpi=300)
    plt.close()
    print("PNG generated: navigator_energy_colored.png")

def generate_interactive_energy_surface():
    X = np.linspace(-3, 3, 200)
    Y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(X, Y)
    Z = np.sin(X) * np.cos(Y) + 0.2 * np.sin(3 * X) + 0.2 * np.cos(3 * Y)
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="Viridis")])
    fig.update_layout(
        title="Interactive 3D Energy Surface",
        scene=dict(
            xaxis_title="Bit-space X",
            yaxis_title="Bit-space Y",
            zaxis_title="Energy"
        ),
        autosize=False,
        width=800,
        height=700
    )
    fig.write_html("interactive_energy_surface.html")
    print("Interactive 3D HTML generated: interactive_energy_surface.html")

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Generating all Borg‑OS visualizations...")
    generate_energy_surface()
    generate_navigator_paths()
    generate_swarm_mesh()
    generate_memory_basins()
    generate_bit_geometry_embedding()
    generate_collapse_dynamics()
    generate_collapse_gif()
    generate_energy_colored_paths()
    generate_interactive_energy_surface()
    print("All visualizations generated successfully.")