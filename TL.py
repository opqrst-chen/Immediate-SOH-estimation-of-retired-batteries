# %%
from PTL.data_process import (
    load_and_process_data,
    filter_data_by_cata_test,
    normalize_features_and_labels,
    stratified_split,
    filter_and_process_data,
    prepare_datasets,
)
from PTL.model import (
    create_soc_estimator,
    create_feature_extractor,
    create_task_net,
    CoralModel,
)
from PTL.utils import (
    set_random_seeds,
    limit_threads,
    coral_loss,
    calculate_mape,
    calculate_maxpe,
    evaluate_soc_predictions,
)
from PTL.evaluator import evaluate_and_plot
from PTL.visualization import plot_losses, plot_feature_distribution
import numpy as np
import tensorflow as tf
from typing import Tuple, List, Any

# %%
# 1. 设置随机种子和限制线程
set_random_seeds()  # 设置随机种子
limit_threads()  # 限制线程数

# 示例数据加载（假设之前的步骤已生成 data）
file_paths: List[str] = [
    "data/processed/combined_augmented_data_output_Cylind21.xlsx",
    "data/processed/combined_augmented_data_output_Pouch31.xlsx",
    "data/processed/combined_augmented_data_output_Pouch52.xlsx",
]

# 加载并合并数据
combined_data: Any = load_and_process_data(file_paths)

# 定义要测试的Cata值和样本比例
Cata_to_test: int = 2
sample_proportion: int = 3

# 调用过滤函数
Fts, SOH, SOC, Cata = filter_data_by_cata_test(
    combined_data, Cata_to_test, sample_proportion
)

# 假设 Fts, SOH, SOC 已经被正确赋值
(Fts_normalized, SOH_normalized, SOC_normalized), (
    feature_scaler,
    label_scaler_SOH,
    SOC_scaler,
) = normalize_features_and_labels(Fts, SOH, SOC)

# 查看结果
print("Normalized Features (Fts):", Fts_normalized.shape)
print("Normalized SOH:", SOH_normalized.shape)
print("Normalized SOC:", SOC_normalized.shape)

# %%
# 调用分层划分函数
X_train, X_test, y_train, y_test, SOC_train, SOC_test, Cata_train, Cata_test = (
    stratified_split(Fts, SOH, SOC, Cata)
)

# 打印划分后的数据形状
print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training SOH Shape:", y_train.shape)
print("Testing SOH Shape:", y_test.shape)
print("Training SOC Shape:", SOC_train.shape)
print("Testing SOC Shape:", SOC_test.shape)
print("Training Cata Shape:", Cata_train.shape)
print("Testing Cata Shape:", Cata_test.shape)

# %%
# 使用 filter_and_process_data 进行数据处理
(
    X_train_Cata1,
    y_train_Cata1,
    SOC_train_Cata1,
    X_test_Cata1,
    y_test_Cata1,
    SOC_test_Cata1,
    X_train_Cata2,
    y_train_Cata2,
    SOC_train_Cata2,
    X_test_Cata2,
    y_test_Cata2,
    SOC_test_Cata2,
) = filter_and_process_data(
    X_train,
    y_train,
    SOC_train,
    X_test,
    y_test,
    SOC_test,
    Cata_train,
    Cata_test,
    Cata_to_test,
    sample_proportion,
)

# %%
# 创建SOC估计模型，输入维度为21
soc_estimator = create_soc_estimator(input_dim=21)

# 创建特征提取器，输入维度为21
feature_extractor = create_feature_extractor(input_dim=21)

# 创建回归任务网络，输入维度为21
task_net = create_task_net(input_dim=21)

# 创建并编译模型
model = CoralModel(soc_estimator, feature_extractor, task_net)
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer)

# %%
# 准备训练数据
print("##### DEBUG #####")
print(
    f"X_train_Cata1: {X_train_Cata1.shape},\n SOC_train_Cata1: {SOC_train_Cata1.shape},\n y_train_Cata1: {y_train_Cata1.shape},\n X_train_Cata2: {X_train_Cata2.shape},\n SOC_train_Cata2: {SOC_train_Cata2.shape},\n y_train_Cata2: {y_train_Cata2.shape}"
)
dataset_combined = prepare_datasets(
    X_train_Cata1,
    SOC_train_Cata1,
    y_train_Cata1,
    X_train_Cata2,
    SOC_train_Cata2,
    y_train_Cata2,
    batch_size=128,
)
print("##### DEBUG #####")

# %%
# 用于记录训练过程中的各项损失
total_loss_list: List[float] = []
task_loss_source_list: List[float] = []
task_loss_target_list: List[float] = []
coral_loss_list: List[float] = []
soc_loss_source_list: List[float] = []
soc_loss_target_list: List[float] = []

# Train model
for epoch in range(50):  # 训练50个epoch
    for batch in dataset_combined:
        data_source, data_target = batch
        x_source, soc_source, y_source = data_source
        x_target, soc_target, y_target = data_target

        loss_metrics = model.train_step(
            (x_source, soc_source, y_source, x_target, soc_target, y_target)
        )

    print(
        f"Epoch {epoch}: Loss = {loss_metrics['loss']}, SOC Loss Source = {loss_metrics['soc_loss_source']}, SOC Loss Target = {loss_metrics['soc_loss_target']}, Task Loss Source = {loss_metrics['task_loss_source']}, Task Loss Target = {loss_metrics['task_loss_target']}, Coral Loss = {loss_metrics['coral_loss']}"
    )

    # Append the losses
    total_loss_list.append(loss_metrics["loss"])
    task_loss_source_list.append(loss_metrics["task_loss_source"])
    task_loss_target_list.append(loss_metrics["task_loss_target"])
    coral_loss_list.append(loss_metrics["coral_loss"])
    soc_loss_source_list.append(loss_metrics["soc_loss_source"])
    soc_loss_target_list.append(loss_metrics["soc_loss_target"])

# %%
# 对源领域进行评估
evaluate_and_plot(
    model,
    X_test_Cata1,
    y_test_Cata1,
    label_scaler_SOH,
    SOC_scaler,
    soc_estimator=model.soc_estimator,
    domain_name="Source Domain",
)

# 对目标领域进行评估
evaluate_and_plot(
    model,
    X_test_Cata2,
    y_test_Cata2,
    label_scaler_SOH,
    SOC_scaler,
    soc_estimator=model.soc_estimator,
    domain_name="Target Domain",
)

# %%
# 评估源域SOC
mape_soc_source, maxpe_soc_source = evaluate_soc_predictions(
    model, X_test_Cata1, SOC_test_Cata1, SOC_scaler, domain="Source"
)

# 评估目标域SOC
mape_soc_target, maxpe_soc_target = evaluate_soc_predictions(
    model, X_test_Cata2, SOC_test_Cata2, SOC_scaler, domain="Target"
)

# %%
# 绘制损失函数图像
plot_losses(
    total_loss_list,
    task_loss_source_list,
    task_loss_target_list,
    coral_loss_list,
    soc_loss_source_list,
    soc_loss_target_list,
)

# 绘制特征分布图像
plot_feature_distribution(model, X_test_Cata1, X_test_Cata2, feature_id=0)
