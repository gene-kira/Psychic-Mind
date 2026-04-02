#!/usr/bin/env python3
# borg_os.py — Unified Borg-OS Visualization & Organ System

import importlib, subprocess, sys, os, argparse, time, random
from datetime import datetime

# ---------------------------------------------------------
# AUTOLOADER
# ---------------------------------------------------------
required = ["numpy","matplotlib","networkx","scipy","Pillow","imageio","plotly"]
for lib in required:
    try: importlib.import_module(lib)
    except ImportError:
        print(f"[Autoloader] Installing {lib}")
        subprocess.check_call([sys.executable,"-m","pip","install",lib])

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation
import networkx as nx
from scipy.spatial.distance import cdist
from PIL import Image
import imageio
import plotly.graph_objects as go

# ---------------------------------------------------------
# GLOBAL CONFIG / BORG THEME
# ---------------------------------------------------------
OUTPUT_PREFIX = "borg_"
BORG_BG="#000000"; BORG_FG="#00FF88"; BORG_CYAN="#00FFFF"
BORG_MAGENTA="#FF00FF"; BORG_YELLOW="#FFFF00"

def borg_figure(figsize=(10,7)):
    return plt.figure(figsize=figsize, facecolor=BORG_BG)

def borg_axes_2d(figsize=(10,7)):
    fig=borg_figure(figsize)
    ax=fig.add_subplot(111)
    ax.set_facecolor(BORG_BG)
    for spine in ax.spines.values(): spine.set_color(BORG_FG)
    ax.tick_params(colors=BORG_FG)
    return fig, ax

def borg_axes_3d(figsize=(10,7)):
    fig=borg_figure(figsize)
    ax=fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BORG_BG)
    ax.xaxis.label.set_color(BORG_FG)
    ax.yaxis.label.set_color(BORG_FG)
    ax.zaxis.label.set_color(BORG_FG)
    ax.tick_params(colors=BORG_FG)
    return fig, ax

def borg_savefig(fig,name,dpi=300):
    fname=OUTPUT_PREFIX+name
    fig.savefig(fname,dpi=dpi,facecolor=fig.get_facecolor(),bbox_inches="tight")
    plt.close(fig)
    print(f"[BORG] PNG generated: {fname}")
    return fname

# ---------------------------------------------------------
# BIT-STATE UTILITIES
# ---------------------------------------------------------
def sample_bitstates(num_states=64,bit_length=16,seed=42):
    rng=np.random.default_rng(seed)
    return rng.integers(0,2,size=(num_states,bit_length))

def bit_energy(bits):
    return bits.sum(axis=1)/bits.shape[1]

def embed_bits_2d(bits,seed=0):
    rng=np.random.default_rng(seed)
    proj=rng.normal(size=(bits.shape[1],2))
    coords=bits@proj
    coords=(coords-coords.min(axis=0))/(coords.max(axis=0)-coords.min(axis=0)+1e-9)
    return coords

def embed_bits_3d(bits,seed=0):
    rng=np.random.default_rng(seed)
    proj=rng.normal(size=(bits.shape[1],3))
    coords=bits@proj
    coords=(coords-coords.min(axis=0))/(coords.max(axis=0)-coords.min(axis=0)+1e-9)
    return coords

# ---------------------------------------------------------
# ORGANS
# ---------------------------------------------------------
class OrganRegistry:
    def __init__(self):
        self.organs={}
        self.state={}
    def register(self,name,organ):
        self.organs[name]=organ
        organ.registry=self
        print(f"[OrganRegistry] Registered organ: {name}")
    def get(self,name):
        return self.organs.get(name)
    def broadcast(self,event,payload=None):
        for organ in self.organs.values():
            organ.on_event(event,payload)
    def update_state(self,key,value):
        self.state[key]=value
        self.broadcast("state_update",{key:value})

class BorgOrgan:
    def __init__(self,name): self.name=name; self.registry=None
    def on_event(self,event,payload): pass

class NodeDiscovery(BorgOrgan):
    def __init__(self,name="node_discovery"):
        super().__init__(name); self.nodes={}
    def discover(self,node_id):
        self.nodes[node_id]={"timestamp":time.time(),"status":"alive"}
        print(f"[NodeDiscovery] Node discovered: {node_id}")
        self.registry.update_state("nodes",self.nodes)
    def heartbeat(self,node_id):
        if node_id in self.nodes:
            self.nodes[node_id]["timestamp"]=time.time()
            print(f"[NodeDiscovery] Heartbeat from {node_id}")
    def prune(self,timeout=5):
        now=time.time(); dead=[nid for nid,info in self.nodes.items() if now-info["timestamp"]>timeout]
        for nid in dead: print(f"[NodeDiscovery] Node lost: {nid}"); del self.nodes[nid]
        if dead: self.registry.update_state("nodes",self.nodes)

class RLBrain(BorgOrgan):
    def __init__(self,name="rl_brain",actions=5):
        super().__init__(name); self.actions=actions
        self.q_table=np.zeros((100,actions)); self.alpha=0.1; self.gamma=0.9; self.epsilon=0.2
    def choose_action(self,state):
        return random.randint(0,self.actions-1) if random.random()<self.epsilon else np.argmax(self.q_table[state])
    def update(self,state,action,reward,next_state):
        self.q_table[state][action]+=self.alpha*(reward+self.gamma*np.max(self.q_table[next_state])-self.q_table[state][action])
    def run_episode(self,steps=50):
        state=random.randint(0,99)
        for _ in range(steps):
            action=self.choose_action(state)
            next_state=random.randint(0,99)
            reward=-abs(next_state-50)
            self.update(state,action,reward,next_state); state=next_state
        print("[RLBrain] Episode complete")

class MemoryBasins(BorgOrgan):
    def __init__(self,name="memory_basins"): super().__init__(name); self.basins={}
    def add_basin(self,key,vector): self.basins[key]=vector; print(f"[MemoryBasins] Basin added: {key}")
    def nearest(self,vector):
        if not self.basins: return None
        dists={k:np.linalg.norm(v-vector) for k,v in self.basins.items()}
        return min(dists,key=dists.get)

class CollapseEngine(BorgOrgan):
    def collapse(self,vector,basins):
        if not basins: return vector
        nearest=basins.nearest(vector)
        return basins.basins[nearest]

class NavigatorCortex(BorgOrgan):
    def navigate(self,vector):
        basins=self.registry.get("memory_basins")
        collapse=self.registry.get("collapse_engine")
        return collapse.collapse(vector,basins)

class SwarmMesh(BorgOrgan):
    def __init__(self,name="swarm_mesh"): super().__init__(name); self.nodes={}
    def add_node(self,node_id,vector): self.nodes[node_id]=vector; print(f"[SwarmMesh] Node added: {node_id}")
    def nearest(self,vector):
        if not self.nodes: return None
        dists={nid:np.linalg.norm(v-vector) for nid,v in self.nodes.items()}
        return min(dists,key=dists.get)

# ---------------------------------------------------------
# VISUALIZATIONS (same as previous bitstate.py)
# ---------------------------------------------------------
# generate_energy_surface(), generate_navigator_paths(), generate_swarm_mesh(),
# generate_memory_basins(), generate_bit_geometry_embedding(), generate_collapse_dynamics(),
# generate_collapse_gif(), generate_energy_colored_paths(), generate_interactive_energy_surface(),
# generate_navigator_animation(), generate_master_log()
# [All code from the previous snippet can be copied here]

# ---------------------------------------------------------
# AUTONOMOUS MODE
# ---------------------------------------------------------
def autonomous_mode(loop=True, interval_sec=10):
    registry=OrganRegistry()
    rl=RLBrain(); mem=MemoryBasins(); collapse=CollapseEngine()
    nav=NavigatorCortex(); swarm=SwarmMesh(); nodes=NodeDiscovery()
    for organ in [rl,mem,collapse,nav,swarm,nodes]: registry.register(organ.name,organ)
    print("[BORG] Entering fully autonomous mode...")
    try:
        while loop:
            node_id=f"node_{random.randint(1,10)}"; nodes.discover(node_id); nodes.prune(timeout=15)
            rl.run_episode(steps=20)
            bits=sample_bitstates(num_states=5,bit_length=8,seed=random.randint(0,1000))
            for i,b in enumerate(bits): mem.add_basin(f"basin_{random.randint(0,1000)}",b)
            for _ in range(3): nav_vector=nav.navigate(np.random.rand(8))
            swarm.add_node(f"swarm_{random.randint(0,10)}", np.random.rand(2))
            run_all()
            print(f"[BORG] Autonomous loop complete. Sleeping {interval_sec}s\n")
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\n[BORG] Autonomous mode terminated by user.")

# ---------------------------------------------------------
# CLI / MAIN
# ---------------------------------------------------------
def cli():
    parser=argparse.ArgumentParser(description="Borg-OS Unified Organ Visualizer")
    parser.add_argument("--all",action="store_true"); parser.add_argument("--navigator",action="store_true")
    parser.add_argument("--swarm",action="store_true"); parser.add_argument("--memory",action="store_true")
    parser.add_argument("--collapse",action="store_true"); parser.add_argument("--interactive",action="store_true")
    parser.add_argument("--master-log",action="store_true"); parser.add_argument("--autonomous",action="store_true")
    args=parser.parse_args()
    if len(sys.argv)==1:
        print("\n[BORG] Bit-State Visualization Organ")
        print("1) Generate ALL\n2) Navigator Cortex\n3) Swarm Mesh\n4) Memory Basins + Bit Geometry")
        print("5) Collapse Engine (frames + GIF)\n6) Interactive 3D Energy Surface\n7) Master Log PNG\n8) Autonomous Mode")
        choice=input("Select option: ").strip()
        if choice=="1": run_all()
        elif choice=="2": generate_navigator_paths(); generate_energy_colored_paths(); generate_navigator_animation()
        elif choice=="3": generate_swarm_mesh()
        elif choice=="4": generate_memory_basins(); generate_bit_geometry_embedding()
        elif choice=="5": generate_collapse_dynamics(); generate_collapse_gif()
        elif choice=="6": generate_energy_surface(); generate_interactive_energy_surface()
        elif choice=="7": generate_master_log()
        elif choice=="8": autonomous_mode()
        else: print("[BORG] Invalid selection.")
        return
    if args.all: run_all()
    if args.navigator: generate_navigator_paths(); generate_energy_colored_paths(); generate_navigator_animation()
    if args.swarm: generate_swarm_mesh()
    if args.memory: generate_memory_basins(); generate_bit_geometry_embedding()
    if args.collapse: generate_collapse_dynamics(); generate_collapse_gif()
    if args.interactive: generate_energy_surface(); generate_interactive_energy_surface()
    if args.master_log: generate_master_log()
    if args.autonomous: autonomous_mode()

if __name__=="__main__":
    cli()