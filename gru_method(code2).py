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
            # Ensure data is loaded as a list of dictionaries
            self.data = pickle.load(f)
            
    def __len__(self): 
        return len(self.data)
        
    def __getitem__(self, idx):
        # Convert to tensor and ensure float32
        past = torch.tensor(self.data[idx]['past'], dtype=torch.float32)
        future = torch.tensor(self.data[idx]['future'], dtype=torch.float32)
        return past, future

class TrajectoryPredictor(nn.Module):
    def __init__(self):
        super(TrajectoryPredictor, self).__init__()
        self.hidden_size = 64
        self.encoder = nn.GRU(2, self.hidden_size, batch_first=True)
        # Assuming future has 6 points (x,y), so output is 12
        self.decoder = nn.Linear(self.hidden_size, 6 * 2) 

    def forward(self, x):
        # x shape: (batch, seq_len, 2)
        _, h = self.encoder(x)
        
        # h shape is (1, batch, 64). We want the batch dim at the front.
        # Use permute or transpose instead of squeeze(0) for safety
        h_last = h[-1] # Takes the last layer's hidden state: (batch, 64)
        
        out = self.decoder(h_last) 
        return out.view(-1, 6, 2) # Reshape to (batch, 6_steps, 2_coords)

# ======================
# 2. SETUP & DATA LOADING
# ======================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Note: Ensure 'trajectory_dataset.pkl' exists in your directory
try:
    full_dataset = TrajectoryDataset("trajectory_dataset.pkl")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
except FileNotFoundError:
    print("❌ Error: 'trajectory_dataset.pkl' not found. Please check the path.")
    exit()

model = TrajectoryPredictor().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# ======================
# 3. TRAINING LOOP
# ======================
print(f"🚀 Training on {device}...")

for epoch in range(10): 
    model.train()
    train_running_loss = 0
    for past, future in train_loader:
        past, future = past.to(device), future.to(device)
        
        optimizer.zero_grad()
        output = model(past)
        loss = criterion(output, future)
        loss.backward()
        optimizer.step()
        
        train_running_loss += loss.item()
    
    # Validation
    model.eval()
    val_running_loss = 0
    with torch.no_grad():
        for v_past, v_future in val_loader:
            v_past, v_future = v_past.to(device), v_future.to(device)
            v_output = model(v_past)
            v_loss = criterion(v_output, v_future)
            val_running_loss += v_loss.item()
    
    print(f"Epoch {epoch+1:02d} | Train Loss: {train_running_loss/len(train_loader):.5f} | Val Loss: {val_running_loss/len(val_loader):.5f}")

torch.save(model.state_dict(), "trajectory_model.pth")

# ======================
# 4. VISUALIZE
# ======================
model.eval()
with torch.no_grad():
    # Use index 0 from val_set
    test_past, test_future = val_set[0]
    # Prediction: model expects (Batch, Seq, Feat)
    input_tensor = test_past.unsqueeze(0).to(device)
    pred = model(input_tensor).cpu().numpy()[0]
    
    plt.figure(figsize=(8, 5))
    plt.plot(test_past[:,0], test_past[:,1], 'bo-', label="Past")
    plt.plot(test_future[:,0], test_future[:,1], 'go-', label="Actual")
    plt.plot(pred[:,0], pred[:,1], 'ro--', label="Predicted")
    plt.legend()
    plt.title("Trajectory Forecast")
    plt.show()