"""
AUTONOMOUS AI CIVILIZATION (REAL LLM BACKEND)

Supports:
- OpenAI API
- Ollama (local)
- OpenAI-compatible servers (LM Studio, vLLM, llama.cpp)
"""

# =========================
# 🔄 AUTO INSTALL
# =========================
import subprocess, sys
def autoload(p):
    try: __import__(p)
    except: subprocess.check_call([sys.executable,"-m","pip","install",p])

for p in ["numpy","pygame","requests","sentence-transformers","scikit-learn"]:
    autoload(p)

# =========================
# ⚙️ CONFIG
# =========================
LLM_MODE = "ollama"  
# options: "openai", "ollama", "local_api"

OPENAI_API_KEY = "YOUR_KEY_HERE"
LOCAL_API_URL = "http://localhost:8000/v1/chat/completions"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

# =========================
# 📦 IMPORTS
# =========================
import numpy as np, time, random, requests
from collections import defaultdict
import pygame
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 🧠 SMART CACHE
# =========================
class Cache:
    def __init__(self):
        self.c={}
    def get(self,k): return self.c.get(k)
    def put(self,k,v): self.c[k]=v

# =========================
# 🤖 REAL LLM
# =========================
class LLM:
    def __init__(self,cache):
        self.cache=cache

    def call_openai(self,prompt):
        url="https://api.openai.com/v1/chat/completions"
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type":"application/json"
        }
        data={
            "model":"gpt-4o-mini",
            "messages":[{"role":"user","content":prompt}],
            "temperature":0.7
        }
        r=requests.post(url,headers=headers,json=data)
        return r.json()["choices"][0]["message"]["content"]

    def call_ollama(self,prompt):
        r=requests.post(OLLAMA_URL,json={
            "model":MODEL_NAME,
            "prompt":prompt,
            "stream":False
        })
        return r.json()["response"]

    def call_local(self,prompt):
        r=requests.post(LOCAL_API_URL,json={
            "model":MODEL_NAME,
            "messages":[{"role":"user","content":prompt}]
        })
        return r.json()["choices"][0]["message"]["content"]

    def generate(self,prompt):
        k=("llm",prompt[:200])
        c=self.cache.get(k)
        if c: return c

        if LLM_MODE=="openai":
            out=self.call_openai(prompt)
        elif LLM_MODE=="ollama":
            out=self.call_ollama(prompt)
        else:
            out=self.call_local(prompt)

        self.cache.put(k,out)
        return out

# =========================
# 🌳 TREE OF THOUGHT
# =========================
class ToT:
    def __init__(self,llm):
        self.llm=llm

    def solve(self,context):
        thoughts=[]
        for i in range(3):
            thoughts.append(self.llm.generate(context + f"\nOption {i}:"))

        scored=[(len(t),t) for t in thoughts]
        return max(scored)[1]

# =========================
# 🧠 MEMORY
# =========================
class Memory:
    def __init__(self):
        self.events=[]
        self.model=SentenceTransformer("all-MiniLM-L6-v2")
        self.vecs=[]

    def add(self,e):
        self.events.append(e)
        self.vecs.append(self.model.encode(e))

    def recall(self,q):
        if not self.vecs: return ""
        sims=cosine_similarity([self.model.encode(q)],self.vecs)[0]
        return self.events[np.argmax(sims)]

# =========================
# 😊 STATE
# =========================
class State:
    def __init__(self):
        self.emotion={"joy":0.5,"anger":0,"fear":0}
        self.needs={"hunger":0.5,"safety":0.5,"social":0.5}

    def update(self):
        for k in self.needs:
            self.needs[k]+=0.01
        for k in self.emotion:
            self.emotion[k]*=0.95

# =========================
# 💰 ECONOMY
# =========================
class Economy:
    def __init__(self):
        self.prices={"food":5,"wood":2}
        self.market={"food":100,"wood":100}

    def trade(self,agent,res,amt):
        cost=self.prices[res]*amt
        if agent.money>=cost:
            agent.money-=cost
            agent.inventory[res]+=amt
            self.market[res]-=amt

# =========================
# 🌍 WORLD
# =========================
class World:
    def __init__(self):
        self.resources={"food":200,"wood":200}
        self.economy=Economy()
        self.danger=False

    def gather(self,a,res):
        if self.resources[res]>0:
            self.resources[res]-=1
            a.inventory[res]+=1

# =========================
# 🧍 AGENT
# =========================
class Agent:
    def __init__(self,id,llm,world):
        self.id=id
        self.llm=llm
        self.world=world

        self.memory=Memory()
        self.state=State()

        self.money=10
        self.inventory=defaultdict(int)

    def perceive(self):
        self.memory.add(f"danger:{self.world.danger}")

    def decide(self):
        context=f"""
You are an autonomous agent in a world.

Needs:{self.state.needs}
Emotion:{self.state.emotion}
Inventory:{dict(self.inventory)}
Money:{self.money}

Decide next action:
- gather food
- gather wood
- trade
- rest
"""

        return self.llm.generate(context)

    def act(self):
        d=self.decide()

        if "food" in d:
            self.world.gather(self,"food")
        elif "wood" in d:
            self.world.gather(self,"wood")
        elif "trade" in d:
            self.world.economy.trade(self,"food",1)

        self.state.update()
        return d

# =========================
# 👥 CIVILIZATION
# =========================
class Civ:
    def __init__(self,n=5):
        self.world=World()
        self.llm=LLM(Cache())
        self.agents=[Agent(i,self.llm,self.world) for i in range(n)]

    def step(self):
        return [a.act() for a in self.agents]

# =========================
# 📊 UI
# =========================
class UI:
    def __init__(self,civ):
        pygame.init()
        self.screen=pygame.display.set_mode((800,500))
        self.civ=civ

    def run(self):
        clock=pygame.time.Clock()
        running=True
        while running:
            for e in pygame.event.get():
                if e.type==pygame.QUIT:
                    running=False

            self.screen.fill((20,20,30))

            for i,a in enumerate(self.civ.agents):
                x=100+i*100
                hunger=int(a.state.needs["hunger"]*255)
                pygame.draw.circle(self.screen,(hunger,100,200),(x,250),30)

            pygame.display.flip()
            self.civ.step()
            clock.tick(2)

# =========================
# 🚀 RUN
# =========================
if __name__=="__main__":
    civ=Civ(6)
    UI(civ).run()