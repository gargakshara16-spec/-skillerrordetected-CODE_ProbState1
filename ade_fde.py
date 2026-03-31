import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- 1. THE LSTM MODEL ---
class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_steps=12):
        super(TrajectoryLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully Connected layer to map hidden state to (x, y) coordinates
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x: (Batch, 8, 2)
        batch_size = x.size(0)
        
        # Get the output from LSTM
        _, (hidden, _) = self.lstm(x)
        
        # Use the last hidden state to predict the next 12 steps
        # We start from the last known position
        last_pos = x[:, -1:, :]
        predictions = []
        
        current_input = hidden[-1].unsqueeze(1) # Use top layer hidden state
        
        # Autoregressive decoding (simplified for this example)
        # In a real model, you'd use a Decoder LSTM, but a Linear projection works well too:
        out = self.fc(hidden[-1]) # (Batch, 2)
        
        # For simplicity in this script, we'll project the hidden state to all 12 steps at once
        # This is a 'Global' prediction head
        flat_out = nn.Linear(self.hidden_size, self.output_steps * 2).to(x.device)(hidden[-1])
        return last_pos + flat_out.view(batch_size, self.output_steps, 2)

# --- 2. EVALUATION DASHBOARD ---
def run_evaluation(model, val_set, device):
    model.eval()
    idx = np.random.randint(0, len(val_set))
    past, future = val_set[idx]
    
    with torch.no_grad():
        input_data = past.unsqueeze(0).to(device)
        pred = model(input_data).cpu().squeeze(0).numpy()
    
    past, future = past.numpy(), future.numpy()
    errors = np.linalg.norm(pred - future, axis=1)
    ade, fde = np.mean(errors), errors[-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(past[:,0], past[:,1], 'ko-', label="History", alpha=0.3)
    ax1.plot(future[:,0], future[:,1], 'go-', label="True Future")
    ax1.plot(pred[:,0], pred[:,1], 'r--o', label="LSTM Prediction")
    ax1.set_title(f"LSTM Results\nADE: {ade:.4f}m | FDE: {fde:.4f}m")
    ax1.legend(); ax1.axis('equal'); ax1.grid(True)

    ax2.plot(range(1, 13), errors, 'r-o')
    ax2.set_title("Error per Step"); ax2.set_xlabel("Step"); ax2.set_ylabel("Meters")
    
    plt.tight_layout()
    plt.show()

# --- 3. MAIN BLOCK ---
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = TrajectoryLSTM().to(device)
    
    # Create Mock Data (Linear + slight curves)
    mock_data = []
    for _ in range(50):
        t = torch.linspace(0, 1, 20).view(-1, 1)
        path = torch.cat([t, t**2], dim=1) + torch.randn(20, 2)*0.02
        mock_data.append((path[:8], path[8:]))

    # Run the eval
    run_evaluation(model, mock_data, device)