import torch
from captum.attr import IntegratedGradients
from evaluate_gradients.MultiLayerConductance import MultiLayerConductance  
from captum.attr import LayerConductance 

# 1) build a simple model: y = 2⋅x
linear = torch.nn.Linear(1,1, bias=False)
linear.weight.data.fill_(2.0)
model = torch.nn.Sequential(linear)  # single‑layer network

# 2) pick input and baseline
inp      = torch.tensor([[3.0]], requires_grad=True)
baseline = torch.tensor([[0.0]])

# 3) IG on input
ig = IntegratedGradients(model)
ig_attr, ig_delta = ig.attribute(
    inp, baselines=baseline, n_steps=50,
    return_convergence_delta=True
)
print("IG attr:", ig_attr, " δ:", ig_delta)

simple_layer_cond = LayerConductance(model, linear)
lc_attr, lc_delta = simple_layer_cond.attribute(
    inp, baselines=baseline, n_steps=50,
    return_convergence_delta=True
)
print("LC attr:", lc_attr, " δ:", lc_delta)
# 4) Conductance on the *only* layer
cond = MultiLayerConductance(model, linear)
ml_attr, ml_delta = cond.attribute(
    inp, baselines=baseline, n_steps=50,
    return_convergence_delta=True
)
print("MLC attr:", ml_attr, " δ:", ml_delta)

