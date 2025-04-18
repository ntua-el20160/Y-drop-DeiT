import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import torchvision
import torchvision.transforms as transforms
# Assuming that your python path can see the parent directory
import sys
import os

# Insert the parent directory into sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

from datasets import create_subdataset

import matplotlib.pyplot as plt



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
        path_simple = []
        unresampled_out = []

        
        for i in range(B):
            img = x_np[i]  # shape: [C, H, W]
            if baseline is None:
                base = np.zeros_like(img)
            else:
                base = baseline[i].detach().cpu().numpy() if baseline.ndim == 4 else baseline.detach().cpu().numpy()
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
            fig, axarr = plt.subplots(1, self.k + 1, figsize=(3*(self.k+1), 3))

            # We'll display the first "anchor" (e.g., baseline anchor_points[0]) 
            # as the "query" image. You could also show the actual target image "img",
            # depending on which you'd like to see.
            query_img = interp_points[0].reshape(C, H, W)

            axarr[0].imshow(query_img.transpose(1, 2, 0))
            axarr[0].axis('off')
            axarr[0].set_title("Query")

            for nbr_i, nbr_idx in enumerate(indices[0]): 
                # The dataset neighbor in flattened space
                neighbor_flat = self.dataset_flat[nbr_idx]
                neighbor_img = neighbor_flat.reshape(C, H, W)

                axarr[nbr_i+1].imshow(neighbor_img.transpose(1, 2, 0))
                axarr[nbr_i+1].axis('off')
                axarr[nbr_i+1].set_title(f"Nbr {nbr_i+1}")

            plt.tight_layout()
            plt.show()
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
            interp_points = interp_points.reshape(n_steps + 1, C, H, W)
            unresampled_path = path_coords.reshape(-1, C, H, W)
            n_cols = max(n_steps+1, unresampled_path.shape[0])
            fig2, axs2 = plt.subplots(3, n_cols, figsize=(2 * n_cols, 6))

            # Row 0: linear path
            for j in range(n_steps+1):
                ax = axs2[0, j] if n_cols > 1 else axs2[0]
                lin_img = interp_points[j].transpose(1, 2, 0)
                ax.imshow(lin_img)
                ax.axis('off')
                ax.set_title(f"Linear {j}")

            # Row 1: unresampled path (raw shortest path)
            for j in range(unresampled_path.shape[0]):
                ax = axs2[1, j] if n_cols > 1 else axs2[1]
                raw_img = unresampled_path[j].transpose(1, 2, 0)
                ax.imshow(raw_img)
                ax.axis('off')
                ax.set_title(f"Raw {j}")

            # Row 2: final resampled path
            for j in range(n_steps+1):
                ax = axs2[2, j] if n_cols > 1 else axs2[2]
                geo_img = resampled_path[j].transpose(1, 2, 0)
                ax.imshow(geo_img)
                ax.axis('off')
                ax.set_title(f"Resamp {j}")

            plt.tight_layout()
            plt.show()

            paths_out.append(torch.tensor(resampled_path, dtype=x.dtype, device=x.device))
            path_simple.append(torch.tensor(interp_points, dtype=x.dtype, device=x.device))
        
        return torch.stack(paths_out, dim=0), torch.stack(path_simple, dim=0)

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
    

def plot_paths(geodesic_paths: torch.Tensor, linear_paths: torch.Tensor, sample_idx: int = 0):
    """
    Plot the geodesic path vs. the linear path for one sample in a batch.
    Both inputs are tensors of shape [B, n_steps+1, C, H, W].
    """
    import matplotlib.pyplot as plt

    # Get the paths for the selected sample.
    geo = geodesic_paths[sample_idx]  # shape: [n_steps+1, C, H, W]
    lin = linear_paths[sample_idx]    # shape: [n_steps+1, C, H, W]
    n_steps_plus1 = geo.shape[0]

    # Create a figure with 2 rows and n_steps+1 columns.
    fig, axs = plt.subplots(2, n_steps_plus1, figsize=(2 * n_steps_plus1, 4))
    for j in range(n_steps_plus1):
        # Convert the geodesic image to numpy (assume C, H, W -> H, W, C).
        geo_img = geo[j].detach().cpu().numpy().transpose(1, 2, 0)
        lin_img = lin[j].detach().cpu().numpy().transpose(1, 2, 0)
        # Plot geodesic image on top row.
        axs[0, j].imshow(geo_img, interpolation='nearest')
        axs[0, j].axis('off')
        axs[0, j].set_title(f"Geo {j}")
        # Plot linear image on bottom row.
        axs[1, j].imshow(lin_img, interpolation='nearest')
        axs[1, j].axis('off')
        axs[1, j].set_title(f"Lin {j}")
    plt.tight_layout()
    plt.show()
# Example usage:
if __name__ == "__main__":
    # Dummy example: assume images of shape 3x32x32 (e.g., CIFAR) and a dataset of 100 images.
    B, C, H, W = 5, 3, 32, 32
    #x_batch = torch.rand(B, C, H, W)
    #dataset = torch.rand(100, C, H, W)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root="C:/Users/tsets/Desktop/Y-drop/data/cifar", train=True, download=True,
                                                transform=transform_train)
    sub_dataset = create_subdataset(train_dataset, batch_size=10, sub_factor=10, stratified=True)
    def preload_subdataset(subdataset):
        """
        Given a small subdataset (a torch.utils.data.Subset),
        load all (data, target) pairs into memory as a list.
        """
        cached = [subdataset[i] for i in range(len(subdataset))]
        return cached

    # Example: Create a subdataset from your full training set.
    # train_dataset is assumed to be already created by your build_dataset.
    # Example usage:
    # sub_dataset = create_subdataset(train_dataset, batch_size=args.batch_size, sub_factor=10, stratified=True)
    # Then pre-load it:
    dataset = preload_subdataset(sub_dataset)
    x_batch = torch.stack([img[0] for img in dataset[:5]], dim=0)  # shape: [B, C, H, W]
    # Optionally, define a dummy gradient function.
    def dummy_gradient_fn(img: torch.Tensor) -> torch.Tensor:
        # For illustration, return ones with the same shape.
        return torch.ones_like(img)
    
    # Create an instance of the GeodesicPathCalculator.
    calculator = GeodesicPathCalculator(dataset, k=5, device=x_batch.device)
    
    # Compute geodesic paths with 10 interpolation steps.
    geodesic_paths,linear_paths = calculator.compute_geodesic_paths_batch(x_batch, n_steps=10, gradient_fn=None)
    plot_paths(geodesic_paths, linear_paths, sample_idx=0)
    print("Geodesic paths shape:", geodesic_paths.shape)
    print("Linear paths shape:", linear_paths.shape)
    # geodesic_paths will have shape [B, 11, C, H, W]
    print("Geodesic paths shape:", geodesic_paths)
