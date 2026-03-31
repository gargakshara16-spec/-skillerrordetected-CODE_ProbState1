import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import matplotlib.pyplot as plt

# ======================
# 1. DATA & MODEL CLASSES
# ======================
class TrajectoryDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]['past']).float(), \
               torch.from_numpy(self.data[idx]['future']).float()

class TrajectoryPredictor(nn.Module):
    def __init__(self):
        super(TrajectoryPredictor, self).__init__()
        self.encoder = nn.GRU(2, 64, batch_first=True)
        self.decoder = nn.Linear(64, 6 * 2)
    def forward(self, x):
        _, h = self.encoder(x)
        return self.decoder(h.squeeze(0)).view(-1, 6, 2)

# ======================
# 2. SETUP & DATA LOADING
# ======================
# Detect MacBook GPU (MPS) or CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load Dataset
full_dataset = TrajectoryDataset("trajectory_dataset.pkl")
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = TrajectoryPredictor().to(device)
print(f"✅ Model Initialized with {count_parameters(model):,} trainable parameters.")

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# ======================
# 3. TRAINING LOOP (Fixed Indentation)
# ======================
print(f"🚀 Training on {device}...")

for epoch in range(10): 
    model.train()
    train_loss = 0
    for past, future in train_loader:
        past, future = past.to(device), future.to(device)
        optimizer.zero_grad()
        output = model(past)
        loss = criterion(output, future)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation phase (Moved outside the training batch loop)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for v_past, v_future in val_loader:
            v_past, v_future = v_past.to(device), v_future.to(device)
            v_loss = criterion(model(v_past), v_future)
            val_loss += v_loss.item()
    
    # Print status once per epoch
    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.5f} | Val Loss: {val_loss/len(val_loader):.5f}")

# Save the model after 10 epochs
torch.save(model.state_dict(), "trajectory_model.pth")
print("✅ Model saved as trajectory_model.pth")

# ======================
# 4. VISUALIZE A PREDICTION
# ======================
model.eval()
with torch.no_grad():
    # Grab one sample from the validation set
    test_past, test_future = val_set[0]
    # Prediction (Add batch dimension with unsqueeze)
    pred = model(test_past.unsqueeze(0).to(device)).cpu().numpy()[0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(test_past[:,0], test_past[:,1], 'bo-', label="Past (Observed)")
    plt.plot(test_future[:,0], test_future[:,1], 'go-', label="True Future")
    plt.plot(pred[:,0], pred[:,1], 'ro--', label="Model Prediction")
    
    plt.axis('equal') # Keeps the coordinate grid proportional
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title("Trajectory Prediction: Real vs Predicted")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.show()