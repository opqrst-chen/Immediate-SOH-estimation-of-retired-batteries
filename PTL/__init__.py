# __init__.py

# 数据处理模块
from .data_process import (
    load_data,
    augment_data,
    load_and_process_data,
    filter_and_process_data,
    stratified_split,
    filter_data_by_cata_test,
    prepare_datasets,
)

# 模型构建模块
from .model import (
    build_vae,
    sampling,
    create_soc_estimator,
    create_feature_extractor,
    create_task_net,
    CoralModel,
)

# 可视化模块
from .visualization import (
    plot_latent_space,
    plot_histogram_and_kde,
    plot_parity,
    plot_losses,
    plot_feature_distribution,
)

# 评估模块
from .evaluator import compute_kl_divergence, evaluate_model

# 工具函数模块
from .utils import (
    set_random_seeds,
    limit_threads,
    coral_loss,
    calculate_mape,
    calculate_maxpe,
    evaluate_soc_predictions,
)

# 配置参数
from .config import *
