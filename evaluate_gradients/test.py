import torch
import torch.nn as nn
from captum._utils.gradient import compute_layer_gradients_and_eval

# Define a simple MLP with two linear layers
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

# Create the model and input
model = SimpleMLP()
input_tensor = torch.randn(2, 4, requires_grad=True)

# Suppose we want to compute gradients and evaluations for fc1 and fc2
layers = [model.fc1, model.fc2]

# Call the compute_layer_gradients_and_eval function.
# (Here, we assume the function is available in the environment.)
layer_gradients, layer_evals = compute_layer_gradients_and_eval(
    forward_fn=model,                    # The forward function of the model.
    layer=layers,                        # Pass a list of layers.
    inputs=input_tensor,                 # Input tensor.
    additional_forward_args=None,        # No additional arguments.
    target_ind=None,                     # No specific target index (output is scalar).
    device_ids=None,                     # Single-device scenario.
    attribute_to_layer_input=False,      # We attribute to layer outputs.
    grad_kwargs={'retain_graph': True}   # Example gradient kwargs.
)

# Display the shapes of the outputs for each layer.
print("Layer Gradients:")
print(layer_gradients)
for i, grads in enumerate(layer_gradients):
    # Each element is a tuple (even if it contains a single tensor).
    print(f"Layer {i+1} gradient shape: {grads[0].shape}")

print("\nLayer Evaluations:")
print(layer_evals)
for i, evals in enumerate(layer_evals):
    # Similarly, each evaluation is provided as a tuple.
    print(f"Layer {i+1} evaluation shape: {evals[0].shape}")

