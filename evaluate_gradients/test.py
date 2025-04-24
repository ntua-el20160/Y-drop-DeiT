import torch
import torch.nn as nn
from MultiLayerSensitivity import MultiLayerSensitivity

# Define a simple model:
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(4, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 1)  # Final output is a scalar per example

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleMLP()
model.eval()

# Batch of 2 samples, each with 4 features:
input_tensor = torch.randn(2, 4)

# Choose the layers for which to compute sensitivity:
layers = [model.fc1, model.fc2]

ml_sens = MultiLayerSensitivity(
    forward_func=model,  # using model.forward
    layer=layers
)

# Compute sensitivity; since model output is scalar per example, target=None is fine.
sens_results = ml_sens.attribute(input_tensor, target=None)
print("MultiLayerSensitivity Results:")
print(sens_results)
