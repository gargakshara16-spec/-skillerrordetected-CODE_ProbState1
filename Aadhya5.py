import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

# --- ENCAPSULATED AI ENGINE ---
class TrajectoryEngine(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        _, (hn, _) = self.lstm(x) # Using the final hidden state directly
        return self.fc(hn[-1])

    @torch.no_grad()
    def forecast(self, history_tensor, steps=8, noise_level=0.1):
        """Generates a future path using autoregressive rollout."""
        self.eval()
        predictions = []
        current_seq = history_tensor.clone()
        
        for _ in range(steps):
            # Predict + Add variation
            pred = self(current_seq) + torch.randn(1, 2) * noise_level
            predictions.append(pred.numpy()[0])
            
            # Update window: Remove first, append new prediction
            new_point = pred.unsqueeze(1)
            current_seq = torch.cat((current_seq[:, 1:, :], new_point), dim=1)
            
        return np.array(predictions)

# --- UTILS ---
def get_mock_data(n_agents=2):
    return [np.cumsum(np.random.randn(10, 2) * 0.3, axis=0) for _ in range(n_agents)]

def check_collision(p1, p2, dist=1.0):
    # Vectorized distance check for speed
    diff = p1 - p2
    distances = np.linalg.norm(diff, axis=1)
    return np.any(distances < dist)

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Trajectory Pro", layout="wide")
st.title("🛡️ SecurePath: AI Collision Avoidance")

# Sidebar
st.sidebar.header("Control Panel")
num_agents = st.sidebar.select_slider("Active Agents", options=[1, 2, 3, 4, 5], value=2)
horizon = st.sidebar.slider("Forecast Horizon", 5, 20, 10)
safety_margin = st.sidebar.slider("Safety Margin", 0.5, 2.5, 1.0)

# Init Engine
@st.cache_resource
def init_engine():
    return TrajectoryEngine()

engine = init_engine()

# Run Simulation
if st.sidebar.button("Run New Simulation"):
    st.cache_data.clear()

agents = get_mock_data(num_agents)

# Layout
c1, c2 = st.columns([3, 1])

with c1:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#f0f2f6')
    
    future_paths = []
    for i, path in enumerate(agents):
        # Convert to tensor
        input_tensor = torch.FloatTensor(path).unsqueeze(0)
        
        # Get Forecast
        future = engine.forecast(input_tensor, steps=horizon)
        future_paths.append(future)
        
        # Plot
        ax.plot(path[:, 0], path[:, 1], 'o-', label=f"Agent {i} (Past)", alpha=0.4)
        ax.plot(future[:, 0], future[:, 1], 'x--', label=f"Agent {i} (AI Forecast)")
        
    ax.legend()
    st.pyplot(fig)

with c2:
    st.subheader("System Status")
    danger = False
    for i in range(len(future_paths)):
        for j in range(i + 1, len(future_paths)):
            if check_collision(future_paths[i], future_paths[j], safety_margin):
                danger = True
                st.warning(f"⚠️ Risk: Agent {i} & {j}")
    
    if not danger:
        st.success("✅ Airspace Clear")

    # Show raw velocity data
    st.write("**Recent Velocities**")
    for i, a in enumerate(agents):
        vel = np.linalg.norm(a[-1] - a[-2])
        st.text(f"A{i}: {vel:.2f} m/s")