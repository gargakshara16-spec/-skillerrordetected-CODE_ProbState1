import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. The Model (Sequence-to-One)
class TrajectoryModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x shape: (batch, seq_len, 2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 2. Multi-Modal Prediction Function (Your Logic)
def predict_multi(model, seq, steps=5, modes=3):
    model.eval()
    all_preds = []

    for _ in range(modes):
        preds = []
        temp_seq = seq.clone()

        for _ in range(steps):
            with torch.no_grad():
                # We add noise to simulate different "branches" of possibility
                base_pred = model(temp_seq)
                noise = torch.randn_like(base_pred) * 0.1 # Reduced noise for stability
                pred = base_pred + noise
                
                preds.append(pred.detach().numpy()[0])

                # Autoregressive step: Push prediction back into sequence
                new = pred.unsqueeze(1) # shape (1, 1, 2)
                temp_seq = torch.cat((temp_seq[:, 1:, :], new), dim=1)

        all_preds.append(np.array(preds))

    return all_preds

# 3. Setup and Mock Training
if __name__ == "__main__":
    model = TrajectoryModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Create a simple "moving straight" dummy dataset
    # 100 samples, 10 steps, 2 coordinates
    dummy_input = torch.cumsum(torch.randn(100, 10, 2) * 0.1, dim=1)
    dummy_target = dummy_input[:, -1, :] + 0.1 # Target is just "keep moving"

    print("Training briefly to stabilize weights...")
    for _ in range(20):
        optimizer.zero_grad()
        loss = criterion(model(dummy_input[:, :-1, :]), dummy_input[:, -1, :])
        loss.backward()
        optimizer.step()

    # 4. Generate Multi-Modal Forecasts
    # Take one sample sequence of 9 points
    sample_seq = dummy_input[0:1, :9, :] 
    
    modes_count = 3
    forecast_steps = 10
    results = predict_multi(model, sample_seq, steps=forecast_steps, modes=modes_count)

    print(f"\nGenerated {modes_count} different futures for the next {forecast_steps} steps:")
    for i, path in enumerate(results):
        print(f"Mode {i+1} final position: {path[-1]}")

    # Optional: Quick Visualization hint
    # plt.plot(path[:, 0], path[:, 1]) etc.