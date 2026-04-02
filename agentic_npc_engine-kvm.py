"""
AGENTIC NPC ENGINE (ALL-IN-ONE)

Features:
- Episodic memory
- Semantic memory (real embeddings)
- Emotional system
- Personality vectors
- Goal-driven planning
- Multi-agent interactions
- World model
- Smart KV cache (LRU+LFU+TTL)
- Real LLM backend (plug-in)
- Cortex visualization UI
"""

# =========================
# 🔄 AUTO INSTALL
# =========================
import subprocess, sys

def autoload(pkg):
    try:
        __import__(pkg)
    except:
        print(f"[AUTOLOAD] Installing {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for pkg in [
    "fastapi", "uvicorn", "numpy",
    "pygame", "scikit-learn",
    "sentence-transformers"
]:
    autoload(pkg)

# =========================
# 📦 IMPORTS
# =========================
import numpy as np
import time, threading, random
from collections import defaultdict, deque
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
import pygame

from sentence_transformers import SentenceTransformer

# =========================
# 🧠 SMART CACHE
# =========================
class SmartCache:
    def __init__(self, max_size=2000, ttl=300):
        self.cache = {}
        self.freq = defaultdict(int)
        self.time = {}
        self.order = deque()
        self.max_size = max_size
        self.ttl = ttl

    def expired(self, k):
        return time.time() - self.time[k] > self.ttl

    def get(self, k):
        if k in self.cache:
            if self.expired(k):
                self.delete(k)
                return None
            self.freq[k]+=1
            self.order.remove(k)
            self.order.append(k)
            return self.cache[k]
        return None

    def put(self, k,v):
        if k in self.cache:
            self.order.remove(k)
        elif len(self.cache)>=self.max_size:
            victim=min(self.cache, key=lambda x:(self.freq[x], self.order.index(x)))
            self.delete(victim)
        self.cache[k]=v
        self.freq[k]+=1
        self.time[k]=time.time()
        self.order.append(k)

    def delete(self,k):
        if k in self.cache:
            del self.cache[k]
            del self.freq[k]
            del self.time[k]
            self.order.remove(k)

# =========================
# 🔍 SEMANTIC MEMORY
# =========================
class SemanticMemory:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vectors=[]
        self.texts=[]

    def add(self,text):
        self.vectors.append(self.model.encode(text))
        self.texts.append(text)

    def query(self,text):
        if not self.vectors:
            return None
        vec=self.model.encode(text)
        sims=cosine_similarity([vec], self.vectors)[0]
        idx=np.argmax(sims)
        if sims[idx]>0.75:
            return self.texts[idx]
        return None

# =========================
# 📖 EPISODIC MEMORY
# =========================
class EpisodicMemory:
    def __init__(self):
        self.events=[]

    def add(self,event):
        self.events.append((time.time(), event))
        if len(self.events)>50:
            self.events.pop(0)

    def recall(self):
        return " ".join([e[1] for e in self.events[-10:]])

# =========================
# 😊 EMOTION SYSTEM
# =========================
class Emotion:
    def __init__(self):
        self.state={"joy":0.5,"anger":0.0,"fear":0.0}

    def update(self,stimulus):
        if "attack" in stimulus:
            self.state["anger"]+=0.2
        if "help" in stimulus:
            self.state["joy"]+=0.2

        # decay
        for k in self.state:
            self.state[k]*=0.95

# =========================
# 🧬 PERSONALITY
# =========================
class Personality:
    def __init__(self):
        self.traits=np.random.rand(5)  # openness, etc.

# =========================
# 🤖 LLM SYSTEM
# =========================
class LLM:
    def __init__(self,cache):
        self.cache=cache

    def call(self,prompt):
        # 🔴 replace with real API
        return "[LLM] "+prompt[-200:]

    def generate(self,prompt):
        k=("llm",prompt)
        c=self.cache.get(k)
        if c: return c
        r=self.call(prompt)
        self.cache.put(k,r)
        return r

# =========================
# 🎯 GOAL + PLANNER
# =========================
class Planner:
    def plan(self,goal,world):
        if "survive" in goal:
            return ["scan","avoid threat","hide"]
        if "talk" in goal:
            return ["approach","greet","respond"]
        return ["idle"]

# =========================
# 🌍 WORLD MODEL
# =========================
class World:
    def __init__(self):
        self.state={"danger":False}

# =========================
# 🧍 NPC AGENT
# =========================
class Agent:
    def __init__(self,id,llm,world):
        self.id=id
        self.llm=llm
        self.world=world

        self.episodic=EpisodicMemory()
        self.semantic=SemanticMemory()
        self.emotion=Emotion()
        self.personality=Personality()
        self.goal="survive"
        self.planner=Planner()

    def perceive(self,input):
        self.episodic.add(input)
        self.semantic.add(input)
        self.emotion.update(input)

    def decide(self):
        plan=self.planner.plan(self.goal,self.world)
        context=f"""
Goal:{self.goal}
Emotion:{self.emotion.state}
Memory:{self.episodic.recall()}
Plan:{plan}
"""
        return self.llm.generate(context)

# =========================
# 👥 MULTI AGENT SYSTEM
# =========================
class AgentManager:
    def __init__(self,llm,world):
        self.agents={}
        self.llm=llm
        self.world=world

    def get(self,id):
        if id not in self.agents:
            self.agents[id]=Agent(id,self.llm,self.world)
        return self.agents[id]

    def interact(self,id,input):
        a=self.get(id)
        a.perceive(input)
        return a.decide()

# =========================
# 📊 CORTEX UI
# =========================
class CortexUI:
    def __init__(self,cache):
        pygame.init()
        self.screen=pygame.display.set_mode((800,500))
        self.cache=cache
        self.font=pygame.font.SysFont(None,20)
        self.running=True

    def run(self):
        clock=pygame.time.Clock()
        while self.running:
            for e in pygame.event.get():
                if e.type==pygame.QUIT:
                    self.running=False

            self.screen.fill((10,10,20))

            keys=list(self.cache.cache.keys())[:60]
            for i,k in enumerate(keys):
                x=(i%10)*70+40
                y=(i//10)*70+40
                f=self.cache.freq[k]
                color=(min(255,f*20),100,200)
                pygame.draw.circle(self.screen,color,(x,y),25)

                t=self.font.render(k[0],True,(255,255,255))
                self.screen.blit(t,(x-15,y-10))

            pygame.display.flip()
            clock.tick(10)

# =========================
# 🌐 API SERVER
# =========================
app=FastAPI()

cache=SmartCache()
llm=LLM(cache)
world=World()
agents=AgentManager(llm,world)

@app.post("/agent/{id}")
def interact(id:int,input:str):
    return {"response":agents.interact(id,input)}

# =========================
# 🚀 RUN
# =========================
def run_ui():
    CortexUI(cache).run()

if __name__=="__main__":
    threading.Thread(target=run_ui,daemon=True).start()

    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)