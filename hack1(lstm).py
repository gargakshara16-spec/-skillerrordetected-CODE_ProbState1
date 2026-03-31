import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# ======================
# 1. DATA & MODEL
# ======================
class TrajectoryDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]['past']).float(), \
               torch.from_numpy(self.data[idx]['future']).float()

class LSTMTrajectoryPredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_steps=6):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_steps * 2)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out.view(-1, 6, 2)

# ======================
# 2. TRAINING ENGINE
# ======================
def train_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Environment: {device} ---")

    # Load Data
    if not os.path.exists("trajectory_dataset.pkl"):
        print("❌ Error: 'trajectory_dataset.pkl' not found.")
        return
    
    dataset = TrajectoryDataset("trajectory_dataset.pkl")
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset)-train_size])
    loader = DataLoader(train_set, batch_size=64, shuffle=True)

    model = LSTMTrajectoryPredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"🚀 Training on {len(train_set)} samples...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        for i, (past, future) in enumerate(loader):
            past, future = past.to(device), future.to(device)
            optimizer.zero_grad()
            loss = criterion(model(past), future)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if i % 500 == 0:
                print(f"  Batch {i}/{len(loader)} | Loss: {loss.item():.4f}")
        
        print(f"✅ Epoch {epoch+1} Finished. Avg Loss: {total_loss/len(loader):.4f}")

    # Save Model
    torch.save(model.state_dict(), "final_lstm_model.pth")
    return model, val_set, device

# ======================
# 3. EVALUATION & CLEANUP
# ======================
def evaluate_and_plot(model, val_set, device):
    model.eval()
    idx = random.randint(0, len(val_set)-1)
    past_t, future_t = val_set[idx]
    
    with torch.no_grad():
        pred_t = model(past_t.unsqueeze(0).to(device)).cpu().squeeze(0)
    
    past, future, pred = past_t.numpy(), future_t.numpy(), pred_t.numpy()

    # Calculate Hackathon Metrics
    ade = np.mean(np.sqrt(np.sum((pred - future)**2, axis=1))) # Avg error
    fde = np.sqrt(np.sum((pred[-1] - future[-1])**2))          # Final error

    print(f"\n--- Performance Metrics ---")
    print(f"📍 ADE (Avg Displacement Error): {ade:.2f} meters")
    print(f"🏁 FDE (Final Displacement Error): {fde:.2f} meters")

    # Final Presentation Plot
    plt.figure(figsize=(10, 7))
    plt.plot(past[:,0], past[:,1], 'bo-', label="History (2s Observed)", alpha=0.6)
    plt.plot(future[:,0], future[:,1], 'go-', label="Ground Truth (3s Real)")
    plt.plot(pred[:,0], pred[:,1], 'ro--', label="LSTM Prediction", linewidth=2)
    
    plt.title(f"Pedestrian Trajectory Prediction\nADE: {ade:.2f}m | FDE: {fde:.2f}m", fontsize=14)
    plt.xlabel("Relative X (m)")
    plt.ylabel("Relative Y (m)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    
    # Save the result for the slide deck
    plt.savefig("hackathon_result.png")
    print("📸 Result saved as 'hackathon_result.png'")
    plt.show()

if __name__ == "__main__":
    trained_model, val_data, dev = train_model()
    if trained_model:
        evaluate_and_plot(trained_model, val_data, dev)