import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def show_accuracy_dashboard(model, val_set, device):
    model.eval()
    
    # 1. Pick a random sample from the set
    idx = np.random.randint(0, len(val_set))
    past, future = val_set[idx]
    
    with torch.no_grad():
        # Move input to the correct device (MPS/CPU)
        input_data = past.unsqueeze(0).to(device)
        # Model prediction
        pred = model(input_data).cpu().squeeze(0).numpy()
    
    past = past.numpy()
    future = future.numpy()

    # 2. Calculate L2 Errors
    errors = np.linalg.norm(pred - future, axis=1)
    ade = np.mean(errors)
    fde = errors[-1]

    # 3. Create a 2-Panel Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=100)

    # Panel 1: Trajectory Map
    ax1.plot(past[:,0], past[:,1], color='gray', marker='o', label="History", alpha=0.5)
    ax1.plot(future[:,0], future[:,1], 'g-o', label="Ground Truth", markersize=4)
    ax1.plot(pred[:,0], pred[:,1], 'r--o', label="AI Prediction", linewidth=2)
    
    for i in range(len(future)):
        ax1.plot([future[i,0], pred[i,0]], [future[i,1], pred[i,1]], 'k-', alpha=0.2)
        
    ax1.set_title(f"Trajectory Map\nADE: {ade:.2f}m | FDE: {fde:.2f}m")
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Panel 2: Error Over Time
    steps = np.arange(len(errors)) + 1 
    ax2.fill_between(steps, 0, errors, color='red', alpha=0.1)
    ax2.plot(steps, errors, 'r-o')
    ax2.set_title("Error Growth")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Meters")

    plt.tight_layout()
    plt.savefig("accuracy_dashboard.png")
    print("\n✅ Success! Dashboard saved as 'accuracy_dashboard.png'")
    plt.show()

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # Device setup for MacBook Air
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on: {device}")

    # --- UPDATED MOCK MODEL (FIXED DEVICE ERROR) ---
    class SimpleMockModel(nn.Module):
        def forward(self, x):
            # Create noise on the SAME device as the input x
            noise = torch.randn(1, 12, 2, device=x.device) * 0.2
            return x[:, -1:, :] + torch.cumsum(noise, dim=1)

    model = SimpleMockModel().to(device)
    
    # Create fake data
    mock_val_set = []
    for _ in range(10):
        p = torch.cumsum(torch.randn(8, 2) * 0.5, dim=0) 
        f = p[-1:] + torch.cumsum(torch.randn(12, 2) * 0.3, dim=0) 
        mock_val_set.append((p, f))

    # Run the dashboard
    show_accuracy_dashboard(model, mock_val_set, device)