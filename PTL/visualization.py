# PTL/visualization.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, entropy
import pandas as pd
from .config import OUTPUT_DIR
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.special import kl_div
from scipy.stats import entropy

import matplotlib.pyplot as plt
import seaborn as sns


def plot_scatter(SOC, Fts, augmented_SOC, augmented_Fts, show_figure=True):
    # Plotting the original vs augmented data using SOC
    # Adjusting plot colors and settings for visual appeal
    plt.figure(figsize=(4, 4))
    # Scatter plot with better color and transparency for visibility
    plt.scatter(
        SOC,
        Fts[:, 0],
        c="#c5e7e8",
        label="Tested Data",
        s=50,
        alpha=0.7,
        edgecolors="#29b4b6",
    )  # Light blue with black edge
    plt.scatter(
        augmented_SOC,
        augmented_Fts[:, 0],
        c="#fbd2cb",
        label="Generated Data",
        # Orange with transparency and black edge
        s=50,
        alpha=0.1,
        edgecolors="#f0776d",
    )
    # Title and labels with size adjustment
    fontsize = 10
    plt.xlabel("SOC [%]", fontsize=fontsize)
    plt.ylabel("Dimension of Voltage Dynamics U1 [V]", fontsize=fontsize)
    # Axis ticks and legend with size adjustments
    plt.xticks(np.arange(5, 55, 5), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # y limit from 3.4 to 4.2
    plt.ylim(3.4, 4.2)
    plt.legend(fontsize=fontsize)
    # Layout adjustment for better spacing
    plt.tight_layout()
    # save the plot for file_path data to jpg 300 dpi
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(OUTPUT_DIR + "figures" + "tested_vs_generated_data_SOC.jpg", dpi=300)
    if show_figure:
        plt.show()


def plot_latent_space(all_encoded_data, data, show_figure=True):
    # Visualizing the Latent Space with colors based on SOH
    plt.figure(figsize=(4, 4))
    fontsize = 10

    # Extract SOH values (make sure it's a 1D array or Series)
    SOH_values = data["SOH"].values  # Ensure this is a numpy array (1D)

    # Scatter plot for latent space visualization
    scatter = plt.scatter(
        all_encoded_data[:, 0],  # First latent dimension
        all_encoded_data[:, 1],  # Second latent dimension
        c=SOH_values,  # Use SOH values for color
        cmap="viridis",  # Color map
        s=50,  # Size of the points
        alpha=0.7,  # Transparency (optional)
    )

    # Colorbar
    cbar = plt.colorbar(scatter, label="SOH")
    cbar.ax.set_ylabel("SOH", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    # Title and axis labels
    plt.title("Visualization of the Latent Space", fontsize=fontsize)
    plt.xlabel("Latent Dimension 1", fontsize=fontsize)
    plt.ylabel("Latent Dimension 2", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Layout adjustment
    plt.tight_layout()

    # Show the figure if requested
    if show_figure:
        plt.show()


def plot_histogram_and_kde(original_data, augmented_data, n_bins=50):
    """绘制原始数据和增强数据的直方图及KDE曲线，并计算KL散度"""
    # Plot histograms
    plt.figure(figsize=(4, 4))
    counts1, bin_edges1 = np.histogram(original_data, bins=n_bins, density=True)
    plt.hist(original_data, bins=bin_edges1, alpha=0.5, label="Tested", density=True)
    counts2, bin_edges2 = np.histogram(augmented_data, bins=bin_edges1, density=True)
    plt.hist(
        augmented_data, bins=bin_edges1, alpha=0.5, label="Generated", density=True
    )
    # Fit KDE to original and augmented data
    kde_orig = gaussian_kde(original_data)
    kde_augmented = gaussian_kde(augmented_data)
    # Define points for plotting
    x_range = np.linspace(min(bin_edges1), max(bin_edges1), 1000)
    # Calculate densities
    density_orig = kde_orig(x_range)
    density_augmented = kde_augmented(x_range)
    # Overlay the KDE on histogram
    plt.plot(x_range, density_orig, label="KDE-Tested")
    plt.plot(x_range, density_augmented, label="KDE-Generated")
    # Calculate KL divergence
    # Adding small constant for numerical stability
    kl_div = entropy(pk=counts1 + 1e-10, qk=counts2 + 1e-10)

    # Plot title and labels
    # plt.title(f"Feature U1 Distribution - KL Divergence: {kl_div:.4f}")
    plt.xlabel("Dimension of Voltage Dynamics U1 [V]")
    plt.ylabel("Gaussian Kernel Density")
    # y limit from 0 to 10
    plt.ylim(0, 10)
    plt.xlim(3.4, 4.2)
    plt.legend()
    # save the plot for file_path data to jpg 300 dpi
    plt.savefig("outputs/figures/tested_vs_generated_U1_distribution.jpg", dpi=300)
    plt.show()

    return kl_div


def plot_kl_divergences_bar_chart(
    kl_divergences, Fts, output_path="outputs/figures/kl_divergences_bar_chart.jpg"
):
    """
    绘制KL散度的条形图，并输出每个特征的KL散度值。

    参数：
    kl_divergences (list or np.array): 每个特征的KL散度值。
    Fts (np.array): 原始数据矩阵，用于确定特征数量（例如U1, U2, ..., Un）。
    output_path (str): 输出保存图像的路径，默认为"outputs/figures/kl_divergences_bar_chart.jpg"。
    """
    # 绘制条形图
    plt.figure(figsize=(4, 4))
    plt.bar(
        [f"U{i+1}" for i in range(Fts.shape[1])],
        kl_divergences,
        color="gray",
        edgecolor="black",
    )
    plt.ylabel("KL Divergence", fontsize=10)

    # 设置y轴范围
    plt.ylim(0, 1)

    # 设置x轴刻度和y轴刻度字体大小
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)

    # 自动调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(output_path, dpi=300)
    plt.show()

    # 输出KL散度值
    for i, kl in enumerate(kl_divergences):
        print(f"KL Divergence for U{i+1}: {kl:.4f}")


### myTL.py ###


def plot_parity(true_values, predicted_values, title):
    # Function to create a parity plot
    plt.figure(figsize=(8, 8))
    plt.scatter(true_values, predicted_values, c="blue", label="Data", alpha=0.3)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title(title)

    # Draw parity line
    min_val = min(min(true_values), min(predicted_values))
    max_val = max(max(true_values), max(predicted_values))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Parity Line")

    plt.legend()


# 绘制损失函数随训练轮次变化的图像
def plot_losses(
    total_loss_list,
    task_loss_source_list,
    task_loss_target_list,
    coral_loss_list,
    soc_loss_source_list,
    soc_loss_target_list,
):
    plt.figure(figsize=(12, 6))
    plt.plot(total_loss_list, label="Total Loss")
    plt.plot(task_loss_source_list, label="Task Loss Source")
    plt.plot(task_loss_target_list, label="Task Loss Target")
    plt.plot(coral_loss_list, label="Coral Loss")
    plt.plot(soc_loss_source_list, label="SOC Loss Source")
    plt.plot(soc_loss_target_list, label="SOC Loss Target")

    plt.title("Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# 绘制源领域和目标领域特征的分布
def plot_feature_distribution(model, X_test_Cata1, X_test_Cata2, feature_id=0):

    # Extract features from source and target domains after training
    # Extract features
    source_features_transformed = model.feature_extractor.predict(X_test_Cata1)
    target_features_transformed = model.feature_extractor.predict(X_test_Cata2)
    # 检查方差为零的特征并跳过
    if np.var(source_features_transformed[:, feature_id]) == 0:
        print(
            f"Warning: Feature {feature_id} in source domain has 0 variance and will be skipped."
        )
    else:
        sns.kdeplot(
            source_features_transformed[:, feature_id],
            fill=True,
            color="blue",
            label="Source (Transformed)",
        )

    if np.var(target_features_transformed[:, feature_id]) == 0:
        print(
            f"Warning: Feature {feature_id} in target domain has 0 variance and will be skipped."
        )
    else:
        sns.kdeplot(
            target_features_transformed[:, feature_id],
            fill=True,
            color="red",
            label="Target (Transformed)",
        )

    plt.xlabel("Feature Value")
    plt.ylabel("Density")
    plt.title("Density of First Feature: Source vs. Target")
    plt.legend()
    plt.show()
