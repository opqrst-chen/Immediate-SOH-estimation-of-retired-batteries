# PTL/visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import gaussian_kde, entropy
from scipy.special import kl_div
from .config import OUTPUT_DIR
from typing import Optional, Union


def plot_scatter(
    SOC: np.ndarray,
    Fts: np.ndarray,
    augmented_SOC: np.ndarray,
    augmented_Fts: np.ndarray,
    show_figure: bool = True,
) -> None:
    """
    绘制原始数据和增强数据的散点图，基于SOC（State of Charge）进行可视化。

    参数：
    SOC (np.ndarray): 原始数据的SOC值。
    Fts (np.ndarray): 原始数据的特征值，形状为 (n_samples, n_features)。
    augmented_SOC (np.ndarray): 增强数据的SOC值。
    augmented_Fts (np.ndarray): 增强数据的特征值，形状为 (n_samples, n_features)。
    show_figure (bool, optional): 是否展示图形，默认为True。
    """
    plt.figure(figsize=(4, 4))
    plt.scatter(
        SOC,
        Fts[:, 0],
        c="#c5e7e8",
        label="Tested Data",
        s=50,
        alpha=0.7,
        edgecolors="#29b4b6",
    )
    plt.scatter(
        augmented_SOC,
        augmented_Fts[:, 0],
        c="#fbd2cb",
        label="Generated Data",
        s=50,
        alpha=0.1,
        edgecolors="#f0776d",
    )

    fontsize = 10
    plt.xlabel("SOC [%]", fontsize=fontsize)
    plt.ylabel("Dimension of Voltage Dynamics U1 [V]", fontsize=fontsize)
    plt.xticks(np.arange(5, 55, 5), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylim(3.4, 4.2)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(OUTPUT_DIR + "figures" + "tested_vs_generated_data_SOC.jpg", dpi=300)

    if show_figure:
        plt.show()


def plot_latent_space(
    all_encoded_data: np.ndarray, data: pd.DataFrame, show_figure: bool = True
) -> None:
    """
    可视化潜在空间，并基于SOH（State of Health）对数据进行着色。

    参数：
    all_encoded_data (np.ndarray): 编码后的数据，形状为 (n_samples, 2)。
    data (pd.DataFrame): 包含SOH值的原始数据。
    show_figure (bool, optional): 是否展示图形，默认为True。
    """
    plt.figure(figsize=(4, 4))
    fontsize = 10

    # 获取SOH值并确保它是1D数组
    SOH_values = data["SOH"].values

    scatter = plt.scatter(
        all_encoded_data[:, 0],
        all_encoded_data[:, 1],
        c=SOH_values,
        cmap="viridis",
        s=50,
        alpha=0.7,
    )

    cbar = plt.colorbar(scatter, label="SOH")
    cbar.ax.set_ylabel("SOH", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    plt.title("Visualization of the Latent Space", fontsize=fontsize)
    plt.xlabel("Latent Dimension 1", fontsize=fontsize)
    plt.ylabel("Latent Dimension 2", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.tight_layout()

    if show_figure:
        plt.show()


def plot_histogram_and_kde(
    original_data: np.ndarray, augmented_data: np.ndarray, n_bins: int = 50
) -> float:
    """
    绘制原始数据和增强数据的直方图及KDE曲线，并计算KL散度。

    参数：
    original_data (np.ndarray): 原始数据。
    augmented_data (np.ndarray): 增强数据。
    n_bins (int, optional): 直方图的箱子数，默认为50。

    返回：
    float: KL散度值。
    """
    plt.figure(figsize=(4, 4))
    counts1, bin_edges1 = np.histogram(original_data, bins=n_bins, density=True)
    plt.hist(original_data, bins=bin_edges1, alpha=0.5, label="Tested", density=True)

    counts2, bin_edges2 = np.histogram(augmented_data, bins=bin_edges1, density=True)
    plt.hist(
        augmented_data, bins=bin_edges1, alpha=0.5, label="Generated", density=True
    )

    kde_orig = gaussian_kde(original_data)
    kde_augmented = gaussian_kde(augmented_data)

    x_range = np.linspace(min(bin_edges1), max(bin_edges1), 1000)
    density_orig = kde_orig(x_range)
    density_augmented = kde_augmented(x_range)

    plt.plot(x_range, density_orig, label="KDE-Tested")
    plt.plot(x_range, density_augmented, label="KDE-Generated")

    kl_div = entropy(pk=counts1 + 1e-10, qk=counts2 + 1e-10)

    plt.xlabel("Dimension of Voltage Dynamics U1 [V]")
    plt.ylabel("Gaussian Kernel Density")
    plt.ylim(0, 10)
    plt.xlim(3.4, 4.2)
    plt.legend()

    plt.savefig("outputs/figures/tested_vs_generated_U1_distribution.jpg", dpi=300)
    plt.show()

    return kl_div


def plot_kl_divergences_bar_chart(
    kl_divergences: Union[np.ndarray, list],
    Fts: np.ndarray,
    output_path: str = "outputs/figures/kl_divergences_bar_chart.jpg",
) -> None:
    """
    绘制KL散度的条形图，并输出每个特征的KL散度值。

    参数：
    kl_divergences (Union[np.ndarray, list]): 每个特征的KL散度值。
    Fts (np.ndarray): 原始数据矩阵，用于确定特征数量。
    output_path (str, optional): 输出保存图像的路径，默认为"outputs/figures/kl_divergences_bar_chart.jpg"。
    """
    plt.figure(figsize=(4, 4))
    plt.bar(
        [f"U{i+1}" for i in range(Fts.shape[1])],
        kl_divergences,
        color="gray",
        edgecolor="black",
    )
    plt.ylabel("KL Divergence", fontsize=10)
    plt.ylim(0, 1)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.show()

    for i, kl in enumerate(kl_divergences):
        print(f"KL Divergence for U{i+1}: {kl:.4f}")


### myTL.py ###


def plot_parity(
    true_values: np.ndarray, predicted_values: np.ndarray, title: str
) -> None:
    """
    绘制真实值与预测值的对比图。

    参数：
    true_values (np.ndarray): 真实值。
    predicted_values (np.ndarray): 预测值。
    title (str): 图表的标题。
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(true_values, predicted_values, c="blue", label="Data", alpha=0.3)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title(title)

    min_val = min(min(true_values), min(predicted_values))
    max_val = max(max(true_values), max(predicted_values))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Parity Line")

    plt.legend()


def plot_losses(
    total_loss_list: list,
    task_loss_source_list: list,
    task_loss_target_list: list,
    coral_loss_list: list,
    soc_loss_source_list: list,
    soc_loss_target_list: list,
) -> None:
    """
    绘制训练过程中各类损失随轮次变化的图像。

    参数：
    total_loss_list (list): 总损失值列表。
    task_loss_source_list (list): 源领域任务损失值列表。
    task_loss_target_list (list): 目标领域任务损失值列表。
    coral_loss_list (list): CORAL损失值列表。
    soc_loss_source_list (list): 源领域SOC损失值列表。
    soc_loss_target_list (list): 目标领域SOC损失值列表。
    """
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


def plot_feature_distribution(
    model: any,  # 假设model是某个类的实例
    X_test_Cata1: np.ndarray,
    X_test_Cata2: np.ndarray,
    feature_id: int = 0,
) -> None:
    """
    绘制源领域和目标领域特征的分布。

    参数：
    model (any): 训练好的模型。
    X_test_Cata1 (np.ndarray): 源领域测试数据。
    X_test_Cata2 (np.ndarray): 目标领域测试数据。
    feature_id (int, optional): 选择的特征索引，默认为0。
    """
    source_features_transformed = model.feature_extractor.predict(X_test_Cata1)
    target_features_transformed = model.feature_extractor.predict(X_test_Cata2)

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
