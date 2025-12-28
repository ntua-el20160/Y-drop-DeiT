# import json
# import os
# import matplotlib.pyplot as plt
# import argparse
# import json
# import os
# import matplotlib.pyplot as plt
# import numpy as np

# def plot_metrics(json1_path, json2_path, title1, title2, save_dir):

#     os.makedirs(save_dir, exist_ok=True)

#     # ---------------- Load JSON ----------------
#     with open(json1_path, "r") as f:
#         m1 = json.load(f)
#     with open(json2_path, "r") as f:
#         m2 = json.load(f)

#     # ============================================================
#     #                     LAYER METRICS
#     # ============================================================
#     layer_names1 = sorted(m1["layers"].keys(), key=lambda x: int(x.split("_")[1]))
#     layer_names2 = sorted(m2["layers"].keys(), key=lambda x: int(x.split("_")[1]))

#     assert layer_names1 == layer_names2, "Layer keys differ between JSONs."

#     layers = [int(x.split("_")[1]) for x in layer_names1]

#     layer_metrics = [
#         "effective_rank",
#         "participation_ratio",
#         "intrinsic_dimension_twonn",
#         "relu_nonzeros",
#     ]

#     for metric in layer_metrics:
#         y1 = [m1["layers"][L][metric] for L in layer_names1]
#         y2 = [m2["layers"][L][metric] for L in layer_names2]

#         plt.figure(figsize=(10,6))
#         plt.plot(layers, y1, "-o", label=title1)
#         plt.plot(layers, y2, "-o", label=title2)

#         plt.xlabel("Layer Index")
#         plt.ylabel(metric)
#         plt.title(f"Layer {metric}")
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()

#         plt.savefig(os.path.join(save_dir, f"layers_{metric}.png"))
#         plt.close()


#     # ============================================================
#     #                   ATTENTION MEAN METRICS
#     # ============================================================
#     blocks1 = sorted(m1["attn"].keys(), key=lambda x: int(x.split("_")[1]))
#     blocks2 = sorted(m2["attn"].keys(), key=lambda x: int(x.split("_")[1]))

#     assert blocks1 == blocks2, "Attention block keys differ."

#     block_ids = [int(x.split("_")[1]) for x in blocks1]

#     mean_keys = {
#         "per_head_effective_rank":       "mean_effective_rank",
#         "per_head_participation_ratio":  "mean_participation_ratio",
#         "intrinsic_dimension_per_head_twonn": "intrinsic_dimension_mean_twonn",
#         "entropy_per_head":              "entropy_mean",
#     }

#     for metric, mean_key in mean_keys.items():

#         y1 = [m1["attn"][b][mean_key] for b in blocks1]
#         y2 = [m2["attn"][b][mean_key] for b in blocks2]

#         plt.figure(figsize=(10,6))
#         plt.plot(block_ids, y1, "-o", label=title1)
#         plt.plot(block_ids, y2, "-o", label=title2)

#         plt.xlabel("Block Index")
#         plt.ylabel(mean_key)
#         plt.title(f"Attention Mean {metric}")
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()

#         plt.savefig(os.path.join(save_dir, f"attn_mean_{metric}.png"))
#         plt.close()


#     # ============================================================
#     #                PER HEAD ATTENTION METRICS
#     # ============================================================
#     for metric in mean_keys.keys():

#         plt.figure(figsize=(10,6))

#         # Model 1 heads
#         for h in range(3):
#             y = [m1["attn"][b][metric][h] for b in blocks1]
#             plt.plot(block_ids, y, "-o", label=f"{title1} head {h}")

#         # Model 2 heads
#         for h in range(3):
#             y = [m2["attn"][b][metric][h] for b in blocks2]
#             plt.plot(block_ids, y, "--o", label=f"{title2} head {h}")

#         plt.xlabel("Block Index")
#         plt.ylabel(metric)
#         plt.title(f"Per-Head Attention Metric: {metric}")
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()

#         plt.savefig(os.path.join(save_dir, f"attn_heads_{metric}.png"))
#         plt.close()



# if __name__ == "__main__":
#     ap = argparse.ArgumentParser(description="Plot and compare metrics from two JSON files.")
#     ap.add_argument("--json1", type=str, help="Path to first single_batch_metrics.json file.")
#     ap.add_argument("--json2", type=str, help="Path to second single_batch_metrics.json file.")
#     ap.add_argument("--title1", type=str, default="Model 1", help="Title for first model in plots.")
#     ap.add_argument("--title2", type=str, default="Model 2", help="Title for second model in plots.")
#     ap.add_argument("--save-dir", type=str, default="./comparison_plots", help="Directory to save plots.")
#     args = ap.parse_args()

#     plot_metrics(args.json1, args.json2, args.title1, args.title2, args.save_dir)
import json
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy import stats
from datetime import datetime
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

class MetricsComparator:
    """Compare and analyze metrics from two trained models."""
    
    def __init__(self, json1_path, json2_path, title1, title2, save_dir):
        self.json1_path = json1_path
        self.json2_path = json2_path
        self.title1 = title1
        self.title2 = title2
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Load data
        with open(json1_path, "r") as f:
            self.m1 = json.load(f)
        with open(json2_path, "r") as f:
            self.m2 = json.load(f)
        
        # Parse layer info
        self.layer_names1 = sorted(self.m1["layers"].keys(), key=lambda x: int(x.split("_")[1]))
        self.layer_names2 = sorted(self.m2["layers"].keys(), key=lambda x: int(x.split("_")[1]))
        assert self.layer_names1 == self.layer_names2, "Layer keys differ between JSONs."
        
        self.layers = [int(x.split("_")[1]) for x in self.layer_names1]
        self.num_layers = len(self.layers)
        
        # Parse attention block info
        self.blocks1 = sorted(self.m1["attn"].keys(), key=lambda x: int(x.split("_")[1]))
        self.blocks2 = sorted(self.m2["attn"].keys(), key=lambda x: int(x.split("_")[1]))
        assert self.blocks1 == self.blocks2, "Attention block keys differ."
        
        self.block_ids = [int(x.split("_")[1]) for x in self.blocks1]
        self.num_blocks = len(self.block_ids)
        
        # Determine number of attention heads
        first_block = self.blocks1[0]
        self.num_heads = len(self.m1["attn"][first_block]["per_head_effective_rank"])
        
        # Log file
        self.log_file = os.path.join(save_dir, "comparison_analysis.log")
        self.log_lines = []
        
    def log(self, message, print_console=True):
        """Add message to log and optionally print."""
        self.log_lines.append(message)
        if print_console:
            print(message)
    
    def write_log(self):
        """Write accumulated log to file."""
        with open(self.log_file, "w") as f:
            f.write("\n".join(self.log_lines))
        print(f"\n📝 Analysis log saved to: {self.log_file}")
    
    def compute_statistics(self, y1, y2, metric_name):
        """Compute statistical comparison between two metric arrays."""
        y1 = np.array(y1)
        y2 = np.array(y2)
        
        # Basic stats
        mean1, std1 = np.mean(y1), np.std(y1)
        mean2, std2 = np.mean(y2), np.std(y2)
        
        # Difference
        diff = mean2 - mean1
        pct_diff = 100 * diff / (mean1 + 1e-8)
        
        # Statistical test (paired t-test)
        t_stat, p_value = stats.ttest_rel(y1, y2)
        
        # Trend analysis (linear regression slope)
        x = np.arange(len(y1))
        slope1, _ = np.polyfit(x, y1, 1)
        slope2, _ = np.polyfit(x, y2, 1)
        
        return {
            'mean1': mean1,
            'std1': std1,
            'mean2': mean2,
            'std2': std2,
            'diff': diff,
            'pct_diff': pct_diff,
            't_stat': t_stat,
            'p_value': p_value,
            'slope1': slope1,
            'slope2': slope2,
            'is_significant': p_value < 0.05
        }
    
    def analyze_layers(self):
        """Comprehensive analysis of layer metrics."""
        self.log("=" * 80)
        self.log("LAYER METRICS ANALYSIS")
        self.log("=" * 80)
        self.log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Model 1: {self.title1} ({self.json1_path})")
        self.log(f"Model 2: {self.title2} ({self.json2_path})")
        self.log(f"Number of layers: {self.num_layers}\n")
        
        layer_metrics = [
            ("effective_rank", "Effective Rank"),
            ("participation_ratio", "Participation Ratio"),
            ("intrinsic_dimension_twonn", "Intrinsic Dimension"),
            ("relu_nonzeros", "ReLU Non-zeros")
        ]
        
        for metric_key, metric_name in layer_metrics:
            self.log(f"\n{'─' * 80}")
            self.log(f"📊 {metric_name.upper()}")
            self.log('─' * 80)
            
            y1 = [self.m1["layers"][L][metric_key] for L in self.layer_names1]
            y2 = [self.m2["layers"][L][metric_key] for L in self.layer_names2]
            
            stats_dict = self.compute_statistics(y1, y2, metric_name)
            
            self.log(f"\n{self.title1}:")
            self.log(f"  Mean: {stats_dict['mean1']:.2f} ± {stats_dict['std1']:.2f}")
            self.log(f"  Trend: {stats_dict['slope1']:+.3f} per layer")
            
            self.log(f"\n{self.title2}:")
            self.log(f"  Mean: {stats_dict['mean2']:.2f} ± {stats_dict['std2']:.2f}")
            self.log(f"  Trend: {stats_dict['slope2']:+.3f} per layer")
            
            self.log(f"\nComparison:")
            self.log(f"  Difference: {stats_dict['diff']:+.2f} ({stats_dict['pct_diff']:+.1f}%)")
            self.log(f"  Statistical significance: {'YES ✓' if stats_dict['is_significant'] else 'NO ✗'} (p={stats_dict['p_value']:.4f})")
            
            # Winner determination
            if abs(stats_dict['pct_diff']) > 5 and stats_dict['is_significant']:
                winner = self.title2 if stats_dict['diff'] > 0 else self.title1
                self.log(f"  🏆 Winner: {winner}")
            else:
                self.log(f"  ⚖️  Similar performance")
            
            # Depth analysis (early vs late layers)
            early_cutoff = self.num_layers // 3
            late_cutoff = 2 * self.num_layers // 3
            
            early1 = np.mean(y1[:early_cutoff])
            late1 = np.mean(y1[late_cutoff:])
            early2 = np.mean(y2[:early_cutoff])
            late2 = np.mean(y2[late_cutoff:])
            
            drop1 = 100 * (early1 - late1) / (early1 + 1e-8)
            drop2 = 100 * (early2 - late2) / (early2 + 1e-8)
            
            self.log(f"\nDepth Progression:")
            self.log(f"  {self.title1}: Early={early1:.1f}, Late={late1:.1f}, Drop={drop1:+.1f}%")
            self.log(f"  {self.title2}: Early={early2:.1f}, Late={late2:.1f}, Drop={drop2:+.1f}%")
            
            if "intrinsic_dimension" in metric_key:
                if drop1 > 10 and drop2 > 10:
                    self.log(f"  ✓ Both models show good abstraction hierarchy")
                elif drop1 > 10:
                    self.log(f"  🏆 {self.title1} has better abstraction hierarchy")
                elif drop2 > 10:
                    self.log(f"  🏆 {self.title2} has better abstraction hierarchy")
                else:
                    self.log(f"  ⚠️  Warning: Neither model shows clear abstraction hierarchy")
            
            # Check for collapsed layers
            threshold = np.mean(y1 + y2) * 0.3  # 30% of average
            collapsed1 = sum(1 for v in y1 if v < threshold)
            collapsed2 = sum(1 for v in y2 if v < threshold)
            
            if collapsed1 > 0 or collapsed2 > 0:
                self.log(f"\n⚠️  Potential Issues:")
                if collapsed1 > 0:
                    self.log(f"  {self.title1}: {collapsed1} layers below threshold ({threshold:.1f})")
                if collapsed2 > 0:
                    self.log(f"  {self.title2}: {collapsed2} layers below threshold ({threshold:.1f})")
    
    def analyze_attention(self):
        """Comprehensive analysis of attention metrics."""
        self.log("\n\n" + "=" * 80)
        self.log("ATTENTION METRICS ANALYSIS")
        self.log("=" * 80)
        
        mean_keys = {
            "per_head_effective_rank": "mean_effective_rank",
            "per_head_participation_ratio": "mean_participation_ratio",
            "intrinsic_dimension_per_head_twonn": "intrinsic_dimension_mean_twonn",
            "entropy_per_head": "entropy_mean",
        }
        
        metric_names = {
            "per_head_effective_rank": "Effective Rank",
            "per_head_participation_ratio": "Participation Ratio",
            "intrinsic_dimension_per_head_twonn": "Intrinsic Dimension",
            "entropy_per_head": "Entropy",
        }
        
        for metric_key, mean_key in mean_keys.items():
            self.log(f"\n{'─' * 80}")
            self.log(f"📊 {metric_names[metric_key].upper()} (Per-Head Mean)")
            self.log('─' * 80)
            
            y1 = [self.m1["attn"][b][mean_key] for b in self.blocks1]
            y2 = [self.m2["attn"][b][mean_key] for b in self.blocks2]
            
            stats_dict = self.compute_statistics(y1, y2, metric_names[metric_key])
            
            self.log(f"\n{self.title1}:")
            self.log(f"  Mean: {stats_dict['mean1']:.3f} ± {stats_dict['std1']:.3f}")
            
            self.log(f"\n{self.title2}:")
            self.log(f"  Mean: {stats_dict['mean2']:.3f} ± {stats_dict['std2']:.3f}")
            
            self.log(f"\nComparison:")
            self.log(f"  Difference: {stats_dict['diff']:+.3f} ({stats_dict['pct_diff']:+.1f}%)")
            self.log(f"  Statistical significance: {'YES ✓' if stats_dict['is_significant'] else 'NO ✗'} (p={stats_dict['p_value']:.4f})")
            
            # Per-head variance analysis
            head_vars1 = []
            head_vars2 = []
            
            for b in self.blocks1:
                head_vals = self.m1["attn"][b][metric_key]
                head_vars1.append(np.var(head_vals))
            
            for b in self.blocks2:
                head_vals = self.m2["attn"][b][metric_key]
                head_vars2.append(np.var(head_vals))
            
            avg_var1 = np.mean(head_vars1)
            avg_var2 = np.mean(head_vars2)
            
            self.log(f"\nHead Specialization (variance across heads):")
            self.log(f"  {self.title1}: {avg_var1:.4f}")
            self.log(f"  {self.title2}: {avg_var2:.4f}")
            
            if avg_var1 > avg_var2 * 1.2:
                self.log(f"  → {self.title1} has more specialized heads")
            elif avg_var2 > avg_var1 * 1.2:
                self.log(f"  → {self.title2} has more specialized heads")
            else:
                self.log(f"  → Similar head specialization")
    
    def generate_summary_report(self):
        """Generate executive summary."""
        self.log("\n\n" + "=" * 80)
        self.log("EXECUTIVE SUMMARY")
        self.log("=" * 80)
        
        # Compute overall scores
        layer_metrics_keys = ["effective_rank", "participation_ratio", "intrinsic_dimension_twonn"]
        
        wins1 = 0
        wins2 = 0
        
        for metric_key in layer_metrics_keys:
            y1 = [self.m1["layers"][L][metric_key] for L in self.layer_names1]
            y2 = [self.m2["layers"][L][metric_key] for L in self.layer_names2]
            
            mean1 = np.mean(y1)
            mean2 = np.mean(y2)
            
            if mean2 > mean1 * 1.05:  # 5% threshold
                wins2 += 1
            elif mean1 > mean2 * 1.05:
                wins1 += 1
        
        self.log(f"\n🏆 Overall Performance:")
        self.log(f"  {self.title1}: {wins1} metrics favored")
        self.log(f"  {self.title2}: {wins2} metrics favored")
        
        if wins2 > wins1:
            self.log(f"\n✨ Recommended: {self.title2}")
        elif wins1 > wins2:
            self.log(f"\n✨ Recommended: {self.title1}")
        else:
            self.log(f"\n⚖️  Models show comparable performance")
        
        self.log("\n" + "=" * 80)
    
    def plot_layer_metrics_enhanced(self):
        """Enhanced plotting for layer metrics."""
        layer_metrics = [
            ("effective_rank", "Effective Rank", "Higher is better"),
            ("participation_ratio", "Participation Ratio", "Higher is better"),
            ("intrinsic_dimension_twonn", "Intrinsic Dimension", "Should decrease with depth"),
            ("relu_nonzeros", "ReLU Non-zeros", "Sparsity indicator")
        ]
        
        for metric_key, metric_name, description in layer_metrics:
            y1 = np.array([self.m1["layers"][L][metric_key] for L in self.layer_names1])
            y2 = np.array([self.m2["layers"][L][metric_key] for L in self.layer_names2])
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Left plot: Line plot with shaded std
            ax1.plot(self.layers, y1, "-o", label=self.title1, linewidth=2, markersize=6, alpha=0.8)
            ax1.plot(self.layers, y2, "-s", label=self.title2, linewidth=2, markersize=6, alpha=0.8)
            
            # Add mean lines
            ax1.axhline(y=np.mean(y1), color='C0', linestyle='--', alpha=0.5, 
                       label=f'{self.title1} mean: {np.mean(y1):.2f}')
            ax1.axhline(y=np.mean(y2), color='C1', linestyle='--', alpha=0.5,
                       label=f'{self.title2} mean: {np.mean(y2):.2f}')
            
            ax1.set_xlabel("Layer Index", fontsize=12)
            ax1.set_ylabel(metric_name, fontsize=12)
            ax1.set_title(f"{metric_name} Across Layers\n({description})", fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='best', framealpha=0.9)
            
            # Right plot: Difference plot
            diff = y2 - y1
            colors = ['green' if d > 0 else 'red' for d in diff]
            
            ax2.bar(self.layers, diff, color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax2.set_xlabel("Layer Index", fontsize=12)
            ax2.set_ylabel(f"Difference ({self.title2} - {self.title1})", fontsize=12)
            ax2.set_title(f"Layer-by-Layer Difference\nGreen = {self.title2} better", fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"layers_{metric_key}_enhanced.png"), bbox_inches='tight')
            plt.close()
        
        # Combined metrics plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (metric_key, metric_name, _) in enumerate(layer_metrics):
            ax = axes[idx]
            y1 = np.array([self.m1["layers"][L][metric_key] for L in self.layer_names1])
            y2 = np.array([self.m2["layers"][L][metric_key] for L in self.layer_names2])
            
            ax.plot(self.layers, y1, "-o", label=self.title1, linewidth=2, alpha=0.8)
            ax.plot(self.layers, y2, "-s", label=self.title2, linewidth=2, alpha=0.8)
            ax.set_xlabel("Layer Index")
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle("All Layer Metrics Comparison", fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "layers_all_metrics_combined.png"), bbox_inches='tight')
        plt.close()
    
    def plot_attention_metrics_enhanced(self):
        """Enhanced plotting for attention metrics."""
        mean_keys = {
            "per_head_effective_rank": ("mean_effective_rank", "Effective Rank"),
            "per_head_participation_ratio": ("mean_participation_ratio", "Participation Ratio"),
            "intrinsic_dimension_per_head_twonn": ("intrinsic_dimension_mean_twonn", "Intrinsic Dimension"),
            "entropy_per_head": ("entropy_mean", "Entropy"),
        }
        
        # Mean metrics plots
        for metric_key, (mean_key, metric_name) in mean_keys.items():
            y1 = np.array([self.m1["attn"][b][mean_key] for b in self.blocks1])
            y2 = np.array([self.m2["attn"][b][mean_key] for b in self.blocks2])
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Left: Line plot
            ax1.plot(self.block_ids, y1, "-o", label=self.title1, linewidth=2, markersize=6)
            ax1.plot(self.block_ids, y2, "-s", label=self.title2, linewidth=2, markersize=6)
            ax1.axhline(y=np.mean(y1), color='C0', linestyle='--', alpha=0.5)
            ax1.axhline(y=np.mean(y2), color='C1', linestyle='--', alpha=0.5)
            ax1.set_xlabel("Block Index", fontsize=12)
            ax1.set_ylabel(f"Mean {metric_name}", fontsize=12)
            ax1.set_title(f"Attention {metric_name} (Mean Across Heads)", fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Right: Difference
            diff = y2 - y1
            colors = ['green' if d > 0 else 'red' for d in diff]
            ax2.bar(self.block_ids, diff, color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax2.set_xlabel("Block Index", fontsize=12)
            ax2.set_ylabel("Difference", fontsize=12)
            ax2.set_title(f"Block-by-Block Difference", fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"attn_mean_{metric_key}_enhanced.png"), bbox_inches='tight')
            plt.close()
        
        # Per-head plots
        for metric_key, (_, metric_name) in mean_keys.items():
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # Model 1 heads
            for h in range(self.num_heads):
                y = [self.m1["attn"][b][metric_key][h] for b in self.blocks1]
                ax.plot(self.block_ids, y, "-o", label=f"{self.title1} head {h}", 
                       linewidth=2, markersize=5, alpha=0.7)
            
            # Model 2 heads
            for h in range(self.num_heads):
                y = [self.m2["attn"][b][metric_key][h] for b in self.blocks2]
                ax.plot(self.block_ids, y, "--s", label=f"{self.title2} head {h}", 
                       linewidth=2, markersize=5, alpha=0.7)
            
            ax.set_xlabel("Block Index", fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f"Per-Head {metric_name}", fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"attn_heads_{metric_key}_enhanced.png"), 
                       bbox_inches='tight')
            plt.close()
    
    def plot_heatmaps(self):
        """Create heatmaps for better pattern visualization."""
        # Layer metrics heatmap
        layer_metrics = ["effective_rank", "participation_ratio", "intrinsic_dimension_twonn"]
        
        data1 = np.array([[self.m1["layers"][L][m] for L in self.layer_names1] for m in layer_metrics])
        data2 = np.array([[self.m2["layers"][L][m] for L in self.layer_names2] for m in layer_metrics])
        
        # Normalize for better visualization
        data1_norm = (data1 - data1.mean(axis=1, keepdims=True)) / (data1.std(axis=1, keepdims=True) + 1e-8)
        data2_norm = (data2 - data2.mean(axis=1, keepdims=True)) / (data2.std(axis=1, keepdims=True) + 1e-8)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        im1 = ax1.imshow(data1_norm, aspect='auto', cmap='RdYlGn', vmin=-2, vmax=2)
        ax1.set_yticks(range(len(layer_metrics)))
        ax1.set_yticklabels(['Eff. Rank', 'Part. Ratio', 'Intr. Dim.'])
        ax1.set_xlabel('Layer Index')
        ax1.set_title(f'{self.title1}\n(Normalized)', fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='Std. deviations from mean')
        
        im2 = ax2.imshow(data2_norm, aspect='auto', cmap='RdYlGn', vmin=-2, vmax=2)
        ax2.set_yticks(range(len(layer_metrics)))
        ax2.set_yticklabels(['Eff. Rank', 'Part. Ratio', 'Intr. Dim.'])
        ax2.set_xlabel('Layer Index')
        ax2.set_title(f'{self.title2}\n(Normalized)', fontweight='bold')
        plt.colorbar(im2, ax=ax2, label='Std. deviations from mean')
        
        diff = data2_norm - data1_norm
        im3 = ax3.imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
        ax3.set_yticks(range(len(layer_metrics)))
        ax3.set_yticklabels(['Eff. Rank', 'Part. Ratio', 'Intr. Dim.'])
        ax3.set_xlabel('Layer Index')
        ax3.set_title(f'Difference\n({self.title2} - {self.title1})', fontweight='bold')
        plt.colorbar(im3, ax=ax3, label='Difference (std.)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "layer_metrics_heatmap.png"), bbox_inches='tight')
        plt.close()
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("\n🔬 Starting comprehensive metrics analysis...\n")
        
        # Analysis
        self.analyze_layers()
        self.analyze_attention()
        self.generate_summary_report()
        
        # Save log
        self.write_log()
        
        # Plotting
        print("\n📊 Generating enhanced visualizations...\n")
        self.plot_layer_metrics_enhanced()
        self.plot_attention_metrics_enhanced()
        self.plot_heatmaps()
        
        print(f"\n✅ Analysis complete! All outputs saved to: {self.save_dir}")
        print(f"📝 Check {self.log_file} for detailed analysis")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Enhanced metrics comparison with statistical analysis.")
    ap.add_argument("--json1", type=str, required=True, help="Path to first metrics JSON file.")
    ap.add_argument("--json2", type=str, required=True, help="Path to second metrics JSON file.")
    ap.add_argument("--title1", type=str, default="Model 1", help="Title for first model.")
    ap.add_argument("--title2", type=str, default="Model 2", help="Title for second model.")
    ap.add_argument("--save-dir", type=str, default="./comparison_plots", help="Directory to save outputs.")
    args = ap.parse_args()
    
    comparator = MetricsComparator(args.json1, args.json2, args.title1, args.title2, args.save_dir)
    comparator.run_full_analysis()