# 🚀 AI Trajectory Pro: Path Prediction Dashboard

## 📄 Project Overview
This project is an AI-driven solution for predicting agent trajectories developed for the hackathon. It leverages deep learning to forecast future coordinates based on historical movement data, providing a visual interface for collision detection and path analysis.
## 🧠 Model Architecture
The system uses a **Recurrent Neural Network (RNN)** approach, specifically an autoregressive rollout model.
* **Core Engine:** Built using **PyTorch** (`nn.Module`) as seen in `TrajectoryEngine`.
* **Logic:** The model takes a history tensor and generates future points step-by-step using an autoregressive loop.
* **Collision Check:** Vectorized distance calculations (`check_collision`) between predicted points to ensure safety.
## 📊 Dataset Used
The model was trained and evaluated on a **Trajectory Dataset** (`trajectory_dataset.pkl`).
* **Format:** Time-series coordinate data $(x, y)$.
* **Input:** Historical movement paths (tensors) with added noise levels for robustness.
* **Output:** Predicted future coordinates for multiple agents simultaneously.
## ⚙️ Setup & Installation Instructions
To set up this project on your local machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/gargakshara16-spec/-A3S-CODE_ProbState1.git](https://github.com/gargakshara16-spec/-A3S-CODE_ProbState1.git)