import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

import torch

import os

def save_image(plt, path, file_name):

    os.makedirs(path, exist_ok = True)
    plt.savefig(f'{path}/{file_name}.png', dpi=300, bbox_inches='tight')



def visualize_embedding(embedding, label, special_points_groups, title = '', filename = '', compress = True, x_name = '', y_name = '', path = '.'):
    # 示例特征矩阵
    feature_matrix = embedding.cpu().detach().numpy() if isinstance(embedding, torch.Tensor) else embedding

    # 假设你的标签矩阵是这样的
    labels_matrix = label.cpu().detach().numpy()

    # 合并所有数据
    all_data = [feature_matrix] + list(special_points_groups.values())
    all_data_combined = np.vstack(all_data)

    if compress:
        tsne = TSNE(n_components=2, perplexity=150, n_iter=1500, learning_rate=300)
        reduced_all_data = tsne.fit_transform(all_data_combined)
    else:
        reduced_all_data = all_data_combined

    # 使用原始数据的长度来分割降维后的数据
    feature_len = feature_matrix.shape[0]
    reduced_features = reduced_all_data[:feature_len]
    start_idx = feature_len
    reduced_special_groups = {}
    for name, val in special_points_groups.items():
        end_idx = start_idx + len(val)
        reduced_special_groups[name] = reduced_all_data[start_idx:end_idx]
        start_idx = end_idx

    # 创建一个大的画布
    plt.figure(figsize=(12, 8))

    # 绘制所有类别的点
    unique_labels = np.unique(labels_matrix)
    for label in unique_labels:
        indices = np.where(labels_matrix == label)[0]
        plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], 
                    alpha=0.7, linewidths=0.5, edgecolors="k", label=f'Class {str(label)}')

    # 动态处理每组特殊点
    color_palette = ['black', 'red', 'blue', 'green', 'yellow']  # 可以添加更多颜色
    for idx, (name, reduced_special_group) in enumerate(reduced_special_groups.items()):
        plt.scatter(reduced_special_group[:, 0], reduced_special_group[:, 1], 
                    c=color_palette[idx % len(color_palette)], s=500 - idx * 100, marker='*', label=name)
        
    if x_name == '':
        x_name = 't-SNE Component 1'

    if y_name == '':
        y_name = 't-SNE Component 2'

    plt.legend(title="Classes")
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    save_image(plt, f'{path}/visualize_embedding/', filename)
    plt.show()

def boxplot_dataframe(df, showfliers, title, filename, path = '.'):
    labels_col = 'label'
    attributes = [col for col in df.columns if col != labels_col]
    
    unique_labels = sorted(df[labels_col].unique())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Total number of plots = number of unique labels * number of attributes
    positions = range(len(unique_labels) * len(attributes))
    color_map = plt.cm.get_cmap('tab10', len(attributes))
    
    for idx, attribute in enumerate(attributes):
        data = [df[df[labels_col] == lbl][attribute].values for lbl in unique_labels]
        positions_for_attribute = range(idx, len(unique_labels) * len(attributes), len(attributes))
        ax.boxplot(data, positions=positions_for_attribute, patch_artist=True, boxprops=dict(facecolor=color_map(idx)), showfliers=showfliers, labels=[attribute]*len(unique_labels))
    
    ax.set_xticks([i + len(attributes) / 2 - 0.5 for i in range(0, len(unique_labels) * len(attributes), len(attributes))])
    ax.set_xticklabels(unique_labels)
    ax.set_xlabel("Labels")
    ax.set_title(title)
    
    # Create custom legend
    custom_lines = [plt.Line2D([0], [0], color=color_map(idx), lw=2) for idx in range(len(attributes))]
    ax.legend(custom_lines, attributes)

    plt.tight_layout()
    save_image(plt, f'{path}/boxplot_dataframe/', filename)
    plt.show()

def plot_percentage_stacked_bar(y_true, y_pred, bar_width=0.4, edgecolor='black', linewidth=1.5, title = '', prefix = '', filename = '', path = '.'):
    """
    绘制百分比堆叠柱状图
    
    参数:
    - y_true: 真实标签列表或数组
    - y_pred: 预测标签列表或数组
    - bar_width: 柱子宽度
    - edgecolor: 子柱边框颜色
    - linewidth: 子柱边框线宽
    """

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 将混淆矩阵的值转化为百分比
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    n_classes = cm.shape[0]

    # 设置图大小
    plt.figure(figsize=(10, 6))

    # 绘制堆叠柱状图
    bottom = np.zeros(n_classes)
    for i in range(n_classes):
        plt.bar(np.arange(n_classes), cm_percentage[:, i], width=bar_width, bottom=bottom, 
                label=f'{prefix} Class {i}', edgecolor=edgecolor, linewidth=linewidth)
        for j, b in enumerate(bottom):
            if cm_percentage[j, i] > 0:  # 非0%情况才显示
                plt.text(j, b + cm_percentage[j, i] / 2 - 0.05, f"{cm_percentage[j, i]*100:.1f}%", ha='center', va='center', color='white')
        bottom += cm_percentage[:, i]

    plt.xlabel('Actual Label')
    plt.ylabel('Percentage')
    plt.title(title)
    plt.xticks(np.arange(n_classes), [f'Class {i}' for i in range(n_classes)])
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    save_image(plt, f'{path}/plot_percentage_stacked_bar/', filename)
    plt.show()

def visualize_data(x, y, filename, path = '.'):
    """
    使用PCA对数据进行降维并可视化

    Parameters:
    - data: 一个Data类的实例

    Returns:
    无
    """

    # 使用PCA降维
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x)

    plt.figure(figsize=(10, 8))
    
    num_classes = len(np.unique(y))
    palette = sns.color_palette("husl", num_classes)
    
    # 使用seaborn的kdeplot函数绘制等高线，并绘制散点图
    for i in range(num_classes):
        idx = np.where(y == i)[0]
        sns.kdeplot(x=x_pca[idx, 0], y=x_pca[idx, 1], fill=True, color=palette[i], alpha=0.3)
        plt.scatter(x_pca[idx, 0], x_pca[idx, 1], color=palette[i], s=30, label=f"Class {i}")

    plt.xlim([x_pca[:, 0].min() - 1, x_pca[:, 0].max() + 1])
    plt.ylim([x_pca[:, 1].min() - 1, x_pca[:, 1].max() + 1])

    # 修复图例问题，只显示一个代表每个类别的标记
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Classes")
    
    # 为背景加上网格
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.title("Visualization of data using PCA")
    plt.tight_layout()
    save_image(plt, f'{path}/visualize_data_PCA/', filename)
    plt.show()

def plot_lines(total, details, title='Losses Over Epochs', total_label='Total Loss', xlabel='Epoch', ylabel='Loss', filename=' ', path='.'):
    """
    Plots a large scale line graph with total losses and individual loss components over epochs.
    The x-axis represents epoch numbers as positive integers starting from 1.
    Adjustments are made for a large number of epochs for better readability, and the total loss
    line is highlighted.
    
    :param total_losses: A list of total loss values for each epoch.
    :param dict_losses: A list of dictionaries or a single dictionary for each epoch, 
                        each dictionary containing individual loss components.
    """
    # Create a larger plot
    plt.figure(figsize=(10, 5))  # You can adjust the size as necessary

    # Generate a list of epoch numbers starting with 1
    epochs = list(range(1, len(total) + 1))

    # Plot the total losses with highlighted features
    plt.plot(epochs, total, label=f'{total_label}', color='red', linewidth=2, linestyle='-')

    # Check if details is a list of dictionaries or a single dictionary
    if isinstance(details, list) and all(isinstance(d, dict) for d in details):
        # If details is a list of dictionaries, iterate over the first dictionary's keys
        linestyles = ['--', '-.', ':', '-']
        colors = ['blue', 'green', 'orange', 'purple']

        for idx, key in enumerate(details[0]):
            component_loss = [d[key] for d in details]
            plt.plot(epochs, component_loss, label=key, color=colors[idx % len(colors)], linestyle=linestyles[idx % len(linestyles)], markersize=4, alpha=0.7, zorder=3, linewidth=1)
    elif isinstance(details, dict):
        # If details is a dictionary, iterate over its keys
        linestyles = ['--', '-.', ':', '-']
        colors = ['blue', 'green', 'orange', 'purple']

        for idx, (key, values) in enumerate(details.items()):
            if isinstance(values, list) and len(values) == len(epochs):
                plt.plot(epochs, values, label=key, color=colors[idx % len(colors)], linestyle=linestyles[idx % len(linestyles)], markersize=4, alpha=0.7, zorder=3, linewidth=1)
            else:
                raise ValueError("Each value in the details dictionary must be a list with length equal to the number of epochs.")

    # Set x-axis ticks to be more sparse
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))  # Adjust the interval as needed

    # Set the size of the ticks labels
    plt.xticks(fontsize=8)  # Adjust the font size as needed
    plt.yticks(fontsize=8)  # Adjust the font size as needed

    # Add legend, title, and axis labels with adjusted size
    plt.legend()
    plt.title(f'{title}', fontsize=10)
    plt.xlabel(f'{xlabel}', fontsize=9)
    plt.ylabel(f'{ylabel}', fontsize=9)

    # Show the plot with a layout that fits the window
    plt.tight_layout()

    # Save the figure if needed
    save_image(plt, f'{path}/plot_lines/', filename)  # Saves a high-resolution version of the plot

    # plt.show()