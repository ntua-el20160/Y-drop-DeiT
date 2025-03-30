import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx

def split_images(self, x: torch.Tensor, n_steps: int = 5) -> torch.Tensor:
    """
    Create interpolation paths from a baseline (zeros) to x.
    Returns a tensor of shape [B, n_steps+1, C, H, W] where B is the batch size.
    """
    baseline = torch.zeros_like(x)  # shape: [B, C, H, W]
    alphas = torch.linspace(0, 1, steps=n_steps + 1, device=x.device).view(1, n_steps + 1, 1, 1, 1)
    x_exp = x.unsqueeze(1)          # shape: [B, 1, C, H, W]
    baseline_exp = baseline.unsqueeze(1)  # shape: [B, 1, C, H, W]
    interpolated = baseline_exp + alphas * (x_exp - baseline_exp)
    return interpolated  # shape: [B, n_steps+1, C, H, W]

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx

class GeodesicPathCalculator:
    def __init__(self, dataset: torch.Tensor, k: int, device: torch.device = None):
        """
        Initialize the calculator by fitting the kNN model on the dataset.
        
        Parameters:
          dataset (torch.Tensor): Tensor of dataset images with shape [N, C, H, W] 
                                  (or [N, D] if pre-flattened).
          k (int): Number of nearest neighbors to use for each interpolation point.
          device (torch.device, optional): Device on which to perform gradient evaluations.
        """
        self.k = k
        self.device = device
        # Convert dataset to numpy array.
        if isinstance(dataset, torch.Tensor):
            dataset_np = dataset.detach().cpu().numpy()
        else:
            dataset_np = np.array(dataset)
        # If dataset images are in [N, C, H, W], flatten them.
        if dataset_np.ndim == 4:
            N = dataset_np.shape[0]
            self.dataset_flat = dataset_np.reshape(N, -1)
        else:
            self.dataset_flat = dataset_np
        # Fit the kNN model on the flattened dataset.
        self.nbrs_dataset = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(self.dataset_flat)

    @staticmethod
    def resample_geodesic_path(path_points: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Given a sequence of points (shape: [P, D]) along the geodesic,
        resample to exactly n_steps+1 points evenly spaced along the cumulative arc-length.
        """
        diffs = np.diff(path_points, axis=0)
        seg_dists = np.linalg.norm(diffs, axis=1)
        cum_dists = np.insert(np.cumsum(seg_dists), 0, 0)
        total_length = cum_dists[-1]
        target_dists = np.linspace(0, total_length, n_steps + 1)
        resampled = np.zeros((n_steps + 1, path_points.shape[1]))
        for d in range(path_points.shape[1]):
            resampled[:, d] = np.interp(target_dists, cum_dists, path_points[:, d])
        return resampled

    def compute_geodesic_path(self, x: torch.Tensor,
                              n_steps: int,
                              gradient_fn=None,
                              grad_steps: int = 10,
                              baseline: torch.Tensor = None) -> torch.Tensor:
        """
        For each image in a batch, compute a geodesic interpolation path from a baseline (default zeros)
        to the image. In this variant, each interpolation point is connected only to its k nearest neighbors
        from the dataset (pre-fitted in the constructor). If the resulting graph is disconnected, a bridging edge 
        is added.
        
        Parameters:
          x           : Tensor of images with shape [B, C, H, W].
          n_steps     : Number of interpolation steps (the resulting path has n_steps+1 points).
          gradient_fn : Optional callable that accepts an image (Tensor of shape [C, H, W]) and returns its gradient.
          grad_steps  : Number of samples to use when approximating the integrated gradient along an edge.
          baseline    : Optional baseline tensor; if None, uses zeros (of same shape as each image).
        
        Returns:
          Tensor of shape [B, n_steps+1, C, H, W] containing the geodesic paths.
        """
        B, C, H, W = x.shape
        x_np = x.detach().cpu().numpy()  # shape: [B, C, H, W]
        paths_out = []
        
        for i in range(B):
            img = x_np[i]  # shape: [C, H, W]
            if baseline is None:
                base = np.zeros_like(img)
            else:
                base = baseline[i].detach().cpu().numpy() if baseline.ndim == 4 else baseline
            # Flatten image and baseline.
            img_flat = img.flatten()
            base_flat = base.flatten()
            
            # Create n_steps+1 interpolation points (anchors) from baseline to image.
            interp_points = np.array([
                base_flat + alpha * (img_flat - base_flat)
                for alpha in np.linspace(0, 1, n_steps + 1)
            ])  # shape: [n_steps+1, D]
            
            # For each interpolation point, find its k nearest neighbors in the dataset.
            distances, indices = self.nbrs_dataset.kneighbors(interp_points)
            # Unique dataset indices that appear as neighbors.
            unique_dataset_indices = np.unique(indices)
            # Map dataset index to a new node ID.
            # Reserve node IDs 0 to n_steps for the interpolation points.
            dataset_node_mapping = {idx: (n_steps + 1 + j) for j, idx in enumerate(unique_dataset_indices)}
            
            # Build the graph: nodes are interpolation points and selected dataset neighbors.
            G = nx.Graph()
            # Add interpolation (anchor) nodes.
            for j in range(n_steps + 1):
                G.add_node(j, coord=interp_points[j])
            # Add dataset neighbor nodes.
            for idx, node_id in dataset_node_mapping.items():
                G.add_node(node_id, coord=self.dataset_flat[idx])
            
            # For each interpolation point, add an edge to each of its k nearest dataset neighbors.
            for j in range(n_steps + 1):
                for neighbor_idx in indices[j]:
                    node_neighbor = dataset_node_mapping[neighbor_idx]
                    p1 = interp_points[j]
                    p2 = self.dataset_flat[neighbor_idx]
                    if gradient_fn is not None:
                        euclid_dist = np.linalg.norm(p1 - p2)
                        t_vals = np.linspace(0, 1, grad_steps)
                        grad_norms = []
                        for t in t_vals:
                            interp_val = p1 + t * (p2 - p1)
                            interp_img = interp_val.reshape((C, H, W))
                            # Evaluate the gradient function on the interpolated image.
                            grad = gradient_fn(torch.tensor(interp_img, dtype=x.dtype, device=self.device))
                            grad_norms.append(torch.norm(grad).item())
                        integrated_norm = np.mean(grad_norms)
                        weight = euclid_dist * integrated_norm
                    else:
                        weight = np.linalg.norm(p1 - p2)
                    G.add_edge(j, node_neighbor, weight=weight)
            
            # If the graph is disconnected between baseline (node 0) and image (node n_steps), add a bridge.
            if not nx.has_path(G, 0, n_steps):
                comps = list(nx.connected_components(G))
                comp_baseline = None
                comp_image = None
                for comp in comps:
                    if 0 in comp:
                        comp_baseline = comp
                    if n_steps in comp:
                        comp_image = comp
                if comp_baseline is not None and comp_image is not None:
                    min_dist = np.inf
                    best_pair = None
                    for u in comp_baseline:
                        for v in comp_image:
                            coord_u = G.nodes[u]['coord']
                            coord_v = G.nodes[v]['coord']
                            d = np.linalg.norm(coord_u - coord_v)
                            if d < min_dist:
                                min_dist = d
                                best_pair = (u, v)
                    if best_pair is not None:
                        u, v = best_pair
                        G.add_edge(u, v, weight=min_dist)
            
            # Compute shortest path from baseline (node 0) to image (node n_steps).
            try:
                path_indices = nx.shortest_path(G, source=0, target=n_steps, weight='weight')
            except nx.NetworkXNoPath:
                path_indices = list(range(n_steps + 1))
            
            path_coords = np.array([G.nodes[node]['coord'] for node in path_indices])
            # Resample to obtain exactly n_steps+1 points (for numerical integration consistency).
            resampled_path = self.resample_geodesic_path(path_coords, n_steps)
            resampled_path = resampled_path.reshape(n_steps + 1, C, H, W)
            paths_out.append(torch.tensor(resampled_path, dtype=x.dtype, device=x.device))
        
        return torch.stack(paths_out, dim=0)

    def compute_geodesic_paths_batch(self, x: torch.Tensor,
                                     n_steps: int,
                                     gradient_fn=None,
                                     grad_steps: int = 10,
                                     baseline: torch.Tensor = None) -> torch.Tensor:
        """
        Compute geodesic paths for a batch of images.
        
        Parameters are the same as in compute_geodesic_path.
        
        Returns:
          Tensor of shape [B, n_steps+1, C, H, W].
        """
        return self.compute_geodesic_path(x, n_steps, gradient_fn=gradient_fn,
                                          grad_steps=grad_steps, baseline=baseline)


# Example usage:
if __name__ == "__main__":
    # Dummy example: assume images of shape 3x32x32 (e.g., CIFAR) and a dataset of 100 images.
    B, C, H, W = 5, 3, 32, 32
    x_batch = torch.rand(B, C, H, W)
    dataset = torch.rand(100, C, H, W)
    
    # Optionally, define a dummy gradient function.
    def dummy_gradient_fn(img: torch.Tensor) -> torch.Tensor:
        # For illustration, return ones with the same shape.
        return torch.ones_like(img)
    
    # Create an instance of the GeodesicPathCalculator.
    calculator = GeodesicPathCalculator(dataset, k=5, device=x_batch.device)
    
    # Compute geodesic paths with 10 interpolation steps.
    geodesic_paths = calculator.compute_geodesic_paths_batch(x_batch, n_steps=10, gradient_fn=None)
    # geodesic_paths will have shape [B, 11, C, H, W]
    print("Geodesic paths shape:", geodesic_paths.shape)
