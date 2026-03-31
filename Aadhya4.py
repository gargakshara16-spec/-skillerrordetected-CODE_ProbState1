import torch
import torch.nn as nn
import numpy as np

# --- 1. MODEL ARCHITECTURE ---
class TrajectoryModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 2. UTILITY & DATA FUNCTIONS ---
def prepare_sequence(coords):
    return torch.tensor(coords, dtype=torch.float32).unsqueeze(0)

def compute_velocity(coords):
    velocities = np.diff(coords, axis=0)
    speed = np.linalg.norm(velocities, axis=1)
    return speed.mean() if len(speed) > 0 else 0

def generate_random_agents(num_agents):
    agents = []
    for _ in range(num_agents):
        x = np.cumsum(np.random.randn(8))
        y = np.cumsum(np.random.randn(8))
        coords = np.stack([x, y], axis=1)
        agents.append(coords)
    return agents

# --- 3. SAFETY & ANALYSIS ---
def detect_collision(traj1, traj2, threshold=1.0):
    """Checks if any two points in the predicted paths are too close."""
    for p1, p2 in zip(traj1, traj2):
        dist = np.linalg.norm(p1 - p2)
        if dist < threshold:
            return True
    return False

# --- 4. MULTI-MODAL PREDICTION ---
def predict_multi(model, seq, steps=5, modes=3):
    model.eval()
    all_preds = []
    for _ in range(modes):
        preds = []
        temp_seq = seq.clone()
        for _ in range(steps):
            with torch.no_grad():
                # Add noise (0.2) to simulate different possible futures
                pred = model(temp_seq) + torch.randn_like(model(temp_seq)) * 0.2
                preds.append(pred.detach().numpy()[0])
                
                # Slide the window: append new prediction, drop oldest point
                new = pred.unsqueeze(1)
                temp_seq = torch.cat((temp_seq[:, 1:, :], new), dim=1)
        all_preds.append(np.array(preds))
    return all_preds

# --- 5. EXECUTION ---
if __name__ == "__main__":
    model = TrajectoryModel()
    agents = generate_random_agents(num_agents=2)
    
    # Predict futures for both agents
    agent1_futures = predict_multi(model, prepare_sequence(agents[0]), steps=5, modes=1)
    agent2_futures = predict_multi(model, prepare_sequence(agents[1]), steps=5, modes=1)

    # Collision check on the first mode of each agent
    path1 = agent1_futures[0]
    path2 = agent2_futures[0]

    if detect_collision(path1, path2, threshold=1.5):
        print("⚠️ Warning: Predicted collision detected between Agent 1 and Agent 2!")
    else:
        print("✅ Paths are clear. No collision predicted.")