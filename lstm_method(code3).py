import torch
import torch.nn as nn
import numpy as np

# 1. Model Definition
class TrajectoryModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 2. Utility Functions (Your logic)
def prepare_sequence(coords):
    # Converts (seq_len, 2) array to (1, seq_len, 2) tensor
    return torch.tensor(coords, dtype=torch.float32).unsqueeze(0)

def compute_velocity(coords):
    # Calculates average speed between coordinate steps
    velocities = np.diff(coords, axis=0)
    speed = np.linalg.norm(velocities, axis=1)
    return speed.mean() if len(speed) > 0 else 0

def generate_random_agents(num_agents):
    agents = []
    for _ in range(num_agents):
        # Brownian motion simulation for 8 time steps
        x = np.cumsum(np.random.randn(8))
        y = np.cumsum(np.random.randn(8))
        coords = np.stack([x, y], axis=1)
        agents.append(coords)
    return agents

# 3. Multi-Modal Prediction (Your logic)
def predict_multi(model, seq, steps=5, modes=3):
    model.eval()
    all_preds = []
    for _ in range(modes):
        preds = []
        temp_seq = seq.clone()
        for _ in range(steps):
            with torch.no_grad():
                # Prediction + small noise to diversify paths
                pred = model(temp_seq) + torch.randn_like(model(temp_seq)) * 0.2
                preds.append(pred.detach().numpy()[0])
                
                # Update sequence: slide window forward with new prediction
                new = pred.unsqueeze(1)
                temp_seq = torch.cat((temp_seq[:, 1:, :], new), dim=1)
        all_preds.append(np.array(preds))
    return all_preds

# 4. Execution Example
if __name__ == "__main__":
    # Initialize
    model = TrajectoryModel()
    agents = generate_random_agents(num_agents=3)
    
    print(f"Processing {len(agents)} agents...")
    
    for i, agent_coords in enumerate(agents):
        # Calculate velocity feature
        avg_speed = compute_velocity(agent_coords)
        
        # Prepare tensor for LSTM
        input_tensor = prepare_sequence(agent_coords)
        
        # Predict 3 possible futures
        futures = predict_multi(model, input_tensor, steps=5, modes=3)
        
        print(f"\nAgent {i+1}:")
        print(f" - Historical Avg Speed: {avg_speed:.4f}")
        print(f" - Mode 1 Final Destination: {futures[0][-1]}")