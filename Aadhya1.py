import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Define the Model Architecture
class TrajectoryModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1):
        super(TrajectoryModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # Predicting next (x, y)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, (hn, cn) = self.lstm(x)
        
        # We only care about the last time step's output for prediction
        last_step_out = out[:, -1, :] 
        return self.fc(last_step_out)

# 2. Generate Synthetic "Hackathon" Data 
# Let's simulate a basic curve: y = x^2 with some noise
def generate_data(num_samples=100, seq_len=10):
    X, Y = [], []
    for _ in range(num_samples):
        # Create a sequence of 10 points
        start_x = np.random.rand()
        x_seq = np.linspace(start_x, start_x + 0.5, seq_len)
        y_seq = x_seq**2 + np.random.normal(0, 0.01, seq_len)
        
        # Input: first 9 points; Target: the 10th point
        X.append(np.stack([x_seq[:-1], y_seq[:-1]], axis=1))
        Y.append([x_seq[-1], y_seq[-1]])
        
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))

# 3. Main Execution Block
if __name__ == "__main__":
    # Hyperparameters
    EPOCHS = 50
    LEARNING_RATE = 0.01
    
    # Initialize Model, Loss, and Optimizer
    model = TrajectoryModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Get Data
    inputs, targets = generate_data()
    
    print("Starting Training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}], Loss: {loss.item():.6f}")

    # 4. Test the model with a fresh sequence
    model.eval()
    test_seq = torch.randn(1, 9, 2) # (1 batch, 9 time steps, 2 coords)
    with torch.no_grad():
        prediction = model(test_seq)
        print("\nTraining Complete!")
        print(f"Input Sequence Shape: {test_seq.shape}")
        print(f"Predicted Next Point (x, y): {prediction.numpy()[0]}")