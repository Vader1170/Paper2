import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
...

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.dpi'] = 150

os.makedirs("../figures", exist_ok=True)

def plot_figure_1():
    """Figure 1: Different mesh families visualised."""
    from mesh_generator import *
    n = 40
    meshes = {
        'Uniform': uniform_mesh(n),
        'Perturbed': perturbed_mesh(n),
        'Clustered': clustered_mesh(n),
        'Geometric': geometric_mesh(n, R=1.2),
        'Exponential': exponential_counterexample(n, R=1.5),
    }
    fig, axes = plt.subplots(len(meshes), 1, figsize=(8, 6), sharex=True)
    for ax, (name, x) in zip(axes, meshes.items()):
        ax.scatter(x, np.ones_like(x), s=20, color='darkblue', alpha=0.8)
        ax.set_ylabel(name, rotation=0, ha='right', va='center', fontsize=10)
        ax.set_yticks([])
        ax.set_xlim(0, 1)
    axes[-1].set_xlabel('x')
    plt.tight_layout()
    plt.savefig("../figures/Figure1_mesh_families.png", dpi=300)
    plt.savefig("../figures/Figure1_mesh_families.pdf")
    plt.close()
    print("Figure 1 saved.")

def plot_figure_2():
    """Figure 2: Condition number vs mesh type."""
    df = pd.read_csv("../data/exp1_kappa_vs_hmax.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x='mesh_type', y='kappa', ax=ax, palette='viridis')
    ax.set_yscale('log')
    ax.set_ylabel(r'Condition number $\kappa_2(G)$')
    ax.set_xlabel('Mesh type')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("../figures/Figure2_kappa_vs_type.png", dpi=300)
    plt.savefig("../figures/Figure2_kappa_vs_type.pdf")
    plt.close()
    print("Figure 2 saved.")

def plot_figure_3():
    """Figure 3: Condition number vs h_max^2."""
    df = pd.read_csv("../data/exp1_kappa_vs_hmax.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, group in df.groupby('mesh_type'):
        ax.scatter(group['h_max']**2, group['kappa'], label=name, alpha=0.6, s=10)
    ax.set_xlabel(r'$h_{\max}^2$')
    ax.set_ylabel(r'$\kappa_2(G)$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    plt.tight_layout()
    plt.savefig("../figures/Figure3_kappa_vs_hmax2.png", dpi=300)
    plt.savefig("../figures/Figure3_kappa_vs_hmax2.pdf")
    plt.close()
    print("Figure 3 saved.")

def plot_figure_4():
    """Figure 4: Balanced Gap comparison (Exp 2)."""
    df = pd.read_csv("../data/exp2_balanced_vs_quasi.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x='n', y='kappa', hue='family', marker='o', ax=ax)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Number of nodes n')
    ax.set_ylabel(r'$\kappa_2(G)$')
    # Add reference line O(1/n^2)
    n_vals = np.sort(df['n'].unique())
    ref_line = df[df['family']=='quasi_uniform'].groupby('n')['kappa'].median().values[0] * (n_vals[0]/n_vals)**2
    ax.plot(n_vals, ref_line, 'k--', label=r'$O(1/n^2)$', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig("../figures/Figure4_balanced_vs_quasi.png", dpi=300)
    plt.savefig("../figures/Figure4_balanced_vs_quasi.pdf")
    plt.close()
    print("Figure 4 saved.")

def plot_figure_5():
    """Figure 5: Failure case (Exp 3)."""
    df = pd.read_csv("../data/exp3_failure.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['n'], df['kappa'], 'o-', color='red', label='Exponential failure')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of nodes n')
    ax.set_ylabel(r'$\kappa_2(G)$')
    ax.legend()
    plt.tight_layout()
    plt.savefig("../figures/Figure5_failure_case.png", dpi=300)
    plt.savefig("../figures/Figure5_failure_case.pdf")
    plt.close()
    print("Figure 5 saved.")

def plot_figure_6():
    """Figure 6: Adaptive weighting improvement (Exp 4)."""
    df = pd.read_csv("../data/exp4_weighting.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x='n', y='kappa', hue='weight_type', marker='o', ax=ax)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Number of nodes n')
    ax.set_ylabel(r'$\kappa_2(G)$')
    ax.legend()
    plt.tight_layout()
    plt.savefig("../figures/Figure6_weighting_comparison.png", dpi=300)
    plt.savefig("../figures/Figure6_weighting_comparison.pdf")
    plt.close()
    print("Figure 6 saved.")

def plot_figure_7():
    """Figure 7: Variance experiment (Exp 5)."""
    df = pd.read_csv("../data/exp5_variance.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['n'], df['var_empirical'], 'o-', label='Empirical variance')
    ax.plot(df['n'], df['bound_theorem4'], 's--', label='Theorem 4 bound (equal)')
    ax.plot(df['n'], df['bound_theorem5'], '^--', label='Theorem 5 bound (adaptive)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of nodes n')
    ax.set_ylabel('Variance of derivative estimate')
    ax.legend()
    plt.tight_layout()
    plt.savefig("../figures/Figure7_variance.png", dpi=300)
    plt.savefig("../figures/Figure7_variance.pdf")
    plt.close()
    print("Figure 7 saved.")

def plot_figure_8():
    """Figure 8: Scaling law (Exp 6)."""
    df = pd.read_csv("../data/exp6_scaling.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    for family, group in df.groupby('family'):
        ax.plot(group['n'], group['kappa'], 'o-', label=family)
    # Add O(1/n^2) reference
    n_ref = np.array([20, 320])
    ref_kappa = 1e-2 * (320/n_ref)**2
    ax.plot(n_ref, ref_kappa, 'k--', label=r'$O(1/n^2)$', alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of nodes n')
    ax.set_ylabel(r'$\kappa_2(G)$')
    ax.legend()
    plt.tight_layout()
    plt.savefig("../figures/Figure8_scaling_law.png", dpi=300)
    plt.savefig("../figures/Figure8_scaling_law.pdf")
    plt.close()
    print("Figure 8 saved.")

def generate_all_figures():
    print("Generating figures...")
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    plot_figure_6()
    plot_figure_7()
    plot_figure_8()
    print("All figures saved to ../figures/")

if __name__ == "__main__":
    generate_all_figures()
