import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ======================
# 1. DEFINE THE ARCHITECTURE (Must match your training file)
# ======================
class TrajectoryPredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_steps=6):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, output_steps * 2)
    def forward(self, x):
        _, h = self.encoder(x)
        return self.decoder(h.squeeze(0)).view(-1, 6, 2)

# ======================
# 2. LOAD THE SAVED BRAIN
# ======================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = TrajectoryPredictor().to(device)

# Load the weights you saved during training
try:
    model.load_state_dict(torch.load("trajectory_model.pth", map_location=device))
    model.eval()
    print("✅ Model weights loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'trajectory_model.pth' not found. Train the model first!")
    exit()

# ======================
# 3. THE PREDICTION FUNCTION
# ======================
def predict_custom_path(coordinates):
    with torch.no_grad():
        # Convert list to Tensor (Batch=1, Seq=4, Dim=2)
        test_input = torch.tensor(coordinates).float().unsqueeze(0).to(device)
        prediction = model(test_input).cpu().squeeze(0).numpy()
        
        # Plotting the "What-If" Scenario
        past = torch.tensor(coordinates).numpy()
        plt.figure(figsize=(8, 6))
        plt.plot(past[:, 0], past[:, 1], 'bo-', label="Your Manual Input")
        plt.plot(prediction[:, 0], prediction[:, 1], 'ro--', label="AI's Prediction")
        
        plt.axhline(0, color='black', lw=0.5, ls='--')
        plt.axvline(0, color='black', lw=0.5, ls='--')
        plt.legend()
        plt.title("Custom Coordinate Test")
        plt.axis('equal')
        plt.grid(True)
        plt.show()
        
        return prediction

# ======================
# 4. MODIFY THESE COORDINATES TO TEST
# ======================
# Example: Person walking 1 meter forward every 0.5 seconds
custom_path = [
    [0.0, 0.0],
    [1.0, 2.5],
    [2.0, 0.0],
    [3.0, 0.0]
]

predict_custom_path(custom_path)