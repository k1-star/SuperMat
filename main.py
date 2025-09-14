import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.patches as mpatches
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, ImageContent, BlobResourceContents
from mcp.types import CallToolRequestParams, CallToolResult, ContentBlock, EmbeddedResource, ResourceLink, Annotations
import logging
import matplotlib
from typing import List, Optional, Tuple, Union, Dict
import base64
from io import BytesIO
import requests
import xml.etree.ElementTree as ET
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, BlobResourceContents
import pandas as pd
import io

import logging
from mp_api.client import MPRester

# 尝试导入 ase 相关模块，如果失败则给出提示
from ase.build import bulk
from ase.calculators.emt import EMT
from tools import computing_tools
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import joblib

# GLOBAL_MODEL = None
# GLOBAL_SCALER = None

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


from mp_api.client import MPRester

dpi = 100  # 图像分辨率

# 配置非交互式图像生成（避免GUI依赖）
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# 全局配置
API_KEY = "57pQ9t5LvE1C4eKzaTBOfFxJbnm3AKqz"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("MPFullVisualizerMCP")
SUPERCON_CSV = "supercon2.csv"  # 请先下载到本地
try:
    df_supercon = pd.read_csv(SUPERCON_CSV)
    logger.info(f"SuperCon dataset loaded: {len(df_supercon)} entries")
except Exception as e:
    logger.error(f"Failed to load SuperCon dataset: {e}")
    df_supercon = pd.DataFrame()
# 初始化 MCP 服务器
mcp = FastMCP()

# 全局存储查询数据（避免重复调用API）
global_query_data: Optional[pd.DataFrame] = None
global_available_fields: List[str] = []

@mcp.tool()
def fetch_materials_data_by_ids(material_ids_list: List[str]) -> TextContent:
    """
    使用 Materials Project API 获取材料数据，并存储为全局 DataFrame 以供后续使用。
    
    参数:
        material_ids: 材料ID列表，例如 ["mp-149", "mp-13"]
    
    返回:
        TextContent: 包含查询结果的文本内容
    """
    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            material_ids = material_ids_list
        )
        if not docs:
            return TextContent(text="未找到指定材料ID的数据。", type="text")
    return TextContent(text=str(docs), type="text")

@mcp.tool()
def fetch_materials_data_by_criteria(elements: List[str], band_gap: Tuple[float, float]) -> TextContent:
    """
    使用 Materials Project API 获取材料数据，并存储为全局 DataFrame 以供后续使用。
    
    参数:
        elements: 元素列表，例如 ["Si", "O"]
        band_gap: 能带隙范围，例如 (0.5, 1.0) 表示 0.5 到 1.0 eV 之间
    返回:
        TextContent: 包含查询结果的文本内容
    """
    global global_query_data, global_available_fields
    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            elements=elements, band_gap=band_gap
        )
        if not docs:
            return TextContent(text="未找到符合条件的材料数据。", type="text")
        
        # 将查询结果转换为 DataFrame
        data = [doc.dict() for doc in docs]
        global_query_data = pd.DataFrame(data)
        global_available_fields = list(global_query_data.columns)
        
    return TextContent(text=f"成功获取 {len(global_query_data)} 条材料数据。可用字段: {', '.join(global_available_fields)}", type="text")


@mcp.tool()
def plot_line(x_data, y_data, labels=None, title="折线图", x_label="X轴", y_label="Y轴", 
              figsize=(10, 6), grid=True, legend=True, style=None):
    """
    绘制折线图
    
    参数:
        x_data: X轴数据，可以是单个数组或数组列表
        y_data: Y轴数据，可以是单个数组或数组列表
        labels: 每条线的标签列表
        title: 图表标题
        x_label: X轴标签
        y_label: Y轴标签
        figsize: 图表大小
        grid: 是否显示网格
        legend: 是否显示图例
        style: 线条样式，如['-', '--', '-.', ':']
    """
    plt.figure(figsize=figsize)
    
    # 确保数据是列表形式，方便统一处理
    if not isinstance(x_data, list):
        x_data = [x_data]
    if not isinstance(y_data, list):
        y_data = [y_data]
    
    # 检查标签数量是否匹配
    if labels is None:
        labels = [f"数据{i+1}" for i in range(len(y_data))]
    elif len(labels) != len(y_data):
        raise ValueError("标签数量与数据系列数量不匹配")
    
    # 检查样式数量
    if style is None:
        style = ['-'] * len(y_data)
    elif len(style) < len(y_data):
        style = style * (len(y_data) // len(style) + 1)
    
    # 绘制每条线
    for x, y, label, fmt in zip(x_data, y_data, labels, style):
        plt.plot(x, y, fmt, label=label)
    
    # 设置图表属性
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    if grid:
        plt.grid(True, linestyle='--', alpha=0.7)
    
    if legend:
        plt.legend()
    
    plt.tight_layout()
        # 将图像保存到内存缓冲区
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    buffer.seek(0)  # 移动到缓冲区开头
    
    # 转换为Base64编码
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # 清理图表，避免内存泄漏
    plt.close()
    
    return ImageContent(data=image_base64, mimeType="image/png", type="image")


@mcp.tool()
def plot_bar(x_data, y_data, labels=None, title="柱状图", x_label="类别", y_label="值",
             figsize=(10, 6), color=None, horizontal=False, grid=False):
    """
    绘制柱状图
    
    参数:
        x_data: X轴类别数据
        y_data: Y轴数值数据，可以是单个数组或数组列表（用于分组柱状图）
        labels: 每组数据的标签（用于分组柱状图）
        title: 图表标题
        x_label: X轴标签
        y_label: Y轴标签
        figsize: 图表大小
        color: 柱子颜色
        horizontal: 是否绘制水平柱状图
        grid: 是否显示网格
    """
    plt.figure(figsize=figsize)
    
    # 处理分组柱状图情况
    if isinstance(y_data[0], (list, np.ndarray)):
        n_groups = len(x_data)
        n_series = len(y_data)
        bar_width = 0.8 / n_series
        
        # 计算每组柱子的位置
        indices = np.arange(n_groups)
        
        for i, (series, label) in enumerate(zip(y_data, labels)):
            position = indices + i * bar_width - 0.4 + bar_width / 2
            if horizontal:
                plt.barh(position, series, bar_width, label=label, color=color[i] if color else None)
                plt.yticks(indices, x_data)
            else:
                plt.bar(position, series, bar_width, label=label, color=color[i] if color else None)
                plt.xticks(indices, x_data)
        
        if labels:
            plt.legend()
    else:
        # 普通柱状图
        if horizontal:
            plt.barh(x_data, y_data, color=color)
        else:
            plt.bar(x_data, y_data, color=color)
    
    # 设置图表属性
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    if grid:
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
        # 将图像保存到内存缓冲区
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    buffer.seek(0)  # 移动到缓冲区开头
    
    # 转换为Base64编码
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # 清理图表，避免内存泄漏
    plt.close()
    
    return ImageContent(data=image_base64, mimeType="image/png", type="image")


@mcp.tool()
def plot_histogram(data, title="直方图", x_label="值", y_label="频数",
                   figsize=(10, 6), color='skyblue', edgecolor='black', density=False):
    """
    绘制直方图
    
    参数:
        data: 要绘制直方图的数据
        title: 图表标题
        x_label: X轴标签
        y_label: Y轴标签
        figsize: 图表大小
        color: 直方图颜色
        edgecolor: 直方图边缘颜色
        density: 是否将直方图标准化为密度图
    """
    bins = 'auto'  
    plt.figure(figsize=figsize)
    
    plt.hist(data, bins=bins, color=color, edgecolor=edgecolor, density=density)
    
    # 设置图表属性
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("密度" if density else y_label)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
        # 将图像保存到内存缓冲区
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    buffer.seek(0)  # 移动到缓冲区开头
    
    # 转换为Base64编码
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # 清理图表，避免内存泄漏
    plt.close()
    
    return ImageContent(data=image_base64, mimeType="image/png", type="image")


@mcp.tool()
def plot_scatter(x_data, y_data, size=None, color=None, cmap='viridis',
                 title="散点图", x_label="X轴", y_label="Y轴", figsize=(10, 6),
                 alpha=0.7, edgecolor='k', colorbar=False):
    """
    绘制散点图
    
    参数:
        x_data: X轴数据
        y_data: Y轴数据
        size: 点的大小，可以是常数或与数据长度相同的数组
        color: 点的颜色，可以是常数或与数据长度相同的数组
        cmap: 颜色映射（当color是数组时）
        title: 图表标题
        x_label: X轴标签
        y_label: Y轴标签
        figsize: 图表大小
        alpha: 点的透明度（0-1之间的数）
        edgecolor: 点的边缘颜色
        colorbar: 是否显示颜色条（当color是数组时）
    """
    plt.figure(figsize=figsize)
    
    scatter = plt.scatter(x_data, y_data, s=size, c=color, cmap=cmap,
                         alpha=float(alpha), edgecolor=edgecolor)
    
    # 设置图表属性
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(linestyle='--', alpha=0.7)
    
    if colorbar and color is not None and not isinstance(color, str):
        plt.colorbar(scatter)
    
    plt.tight_layout()
        # 将图像保存到内存缓冲区
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    buffer.seek(0)  # 移动到缓冲区开头
    
    # 转换为Base64编码
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # 清理图表，避免内存泄漏
    plt.close()
    
    return ImageContent(data=image_base64, mimeType="image/png", type="image")


@mcp.tool()
def plot_pie(data, labels, title="饼图", figsize=(8, 8), autopct='%1.1f%%',
             startangle=90, explode=None, colors=None):
    """
    绘制饼图
    
    参数:
        data: 饼图数据
        labels: 每个部分的标签
        title: 图表标题
        figsize: 图表大小
        autopct: 百分比显示格式
        startangle: 饼图起始角度
        explode: 每个部分的偏移量
        colors: 每个部分的颜色
    """
    plt.figure(figsize=figsize)
    
    plt.pie(data, labels=labels, autopct=autopct, startangle=startangle,
            explode=explode, colors=colors, wedgeprops=dict(edgecolor='w'))
    
    # 设置图表属性
    plt.title(title)
    plt.axis('equal')  # 保证饼图是正圆形
    
    plt.tight_layout()
        # 将图像保存到内存缓冲区
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    buffer.seek(0)  # 移动到缓冲区开头
    
    # 转换为Base64编码
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # 清理图表，避免内存泄漏
    plt.close()
    
    return ImageContent(data=image_base64, mimeType="image/png", type="image")


@mcp.tool()
def plot_boxplot(data, labels=None, title="箱线图", x_label="类别", y_label="值",
                 figsize=(10, 6), color='lightblue', showfliers=True):
    """
    绘制箱线图
    
    参数:
        data: 数据，可以是单个数组或数组列表
        labels: 每个数据集的标签
        title: 图表标题
        x_label: X轴标签
        y_label: Y轴标签
        figsize: 图表大小
        color: 箱体颜色
        showfliers: 是否显示异常值
    """
    plt.figure(figsize=figsize)
    
    # 确保数据是列表形式
    if not isinstance(data, list):
        data = [data]
    
    bp = plt.boxplot(data, labels=labels, patch_artist=True, showfliers=showfliers)
    
    # 设置箱体颜色
    for patch in bp['boxes']:
        patch.set_facecolor(color)
    
    # 设置图表属性
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
        # 将图像保存到内存缓冲区
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    buffer.seek(0)  # 移动到缓冲区开头
    
    # 转换为Base64编码
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # 清理图表，避免内存泄漏
    plt.close()
    
    return ImageContent(data=image_base64, mimeType="image/png", type="image")


@mcp.tool()
def plot_heatmap(data, title="热力图", figsize=(10, 8), cmap='coolwarm',
                 annot=False, fmt='.2f', linewidths=0.5):
    """
    绘制热力图
    
    参数:
        data: 二维数据数组或DataFrame
        title: 图表标题
        figsize: 图表大小
        cmap: 颜色映射
        annot: 是否在单元格中显示数值
        fmt: 数值显示格式
        linewidths: 单元格之间的线宽
    """
    plt.figure(figsize=figsize)
    
    sns.heatmap(data, cmap=cmap, annot=annot, fmt=fmt, linewidths=linewidths)
    
    # 设置图表属性
    plt.title(title)
    
    plt.tight_layout()
        # 将图像保存到内存缓冲区
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    buffer.seek(0)  # 移动到缓冲区开头
    
    # 转换为Base64编码
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # 清理图表，避免内存泄漏
    plt.close()
    
    return ImageContent(data=image_base64, mimeType="image/png", type="image")


@mcp.tool()
def plot_polar(theta, r, title="极坐标图", figsize=(8, 8), color='blue', linewidth=2):
    """
    绘制极坐标图
    
    参数:
        theta: 角度数据（弧度）
        r: 半径数据
        title: 图表标题
        figsize: 图表大小
        color: 线条颜色
        linewidth: 线条宽度
    """
    plt.figure(figsize=figsize)
    ax = plt.subplot(111, polar=True)
    
    ax.plot(theta, r, color=color, linewidth=linewidth)
    ax.fill(theta, r, color=color, alpha=0.25)
    
    # 设置图表属性
    plt.title(title)
    
    plt.tight_layout()
        # 将图像保存到内存缓冲区
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    buffer.seek(0)  # 移动到缓冲区开头
    
    # 转换为Base64编码
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # 清理图表，避免内存泄漏
    plt.close()
    
    return ImageContent(data=image_base64, mimeType="image/png", type="image")


@mcp.tool()
def plot_3d_scatter(x, y, z, color=None, size=50, cmap='viridis',
                    title="3D散点图", x_label="X轴", y_label="Y轴", z_label="Z轴",
                    figsize=(10, 8), alpha=0.7):
    """
    绘制3D散点图
    
    参数:
        x, y, z: 三维数据
        color: 点的颜色
        size: 点的大小
        cmap: 颜色映射
        title: 图表标题
        x_label, y_label, z_label: 三个轴的标签
        figsize: 图表大小
        alpha: 点的透明度（0-1之间的数）
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(x, y, z, c=color, s=size, cmap=cmap, alpha=float(alpha))
    
    # 设置图表属性
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    
    if color is not None and not isinstance(color, str):
        fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    return fig


@mcp.tool()
def plot_surface(x, y, z, cmap='viridis', title="3D曲面图", 
                 x_label="X轴", y_label="Y轴", z_label="Z轴", figsize=(10, 8)):
    """
    绘制3D曲面图
    
    参数:
        x, y: 网格数据
        z: 对应的高度数据
        cmap: 颜色映射
        title: 图表标题
        x_label, y_label, z_label: 三个轴的标签
        figsize: 图表大小
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(x, y, z, cmap=cmap, linewidth=0, antialiased=True)
    
    # 设置图表属性
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    return fig




# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


@mcp.tool()
def plot_violin(data, labels=None, title="小提琴图", x_label="类别", y_label="值",
                figsize=(10, 6), inner="box", split=False, palette=None):
    """
    绘制小提琴图，结合了箱线图和核密度估计的特点
    
    参数:
        data: 数据，可以是单个数组或数组列表
        labels: 每个数据集的标签
        title: 图表标题
        x_label: X轴标签
        y_label: Y轴标签
        figsize: 图表大小
        inner: 小提琴内部显示的内容，可选"box", "quartile", "point", "stick", None
        split: 是否将小提琴分为两半（用于对比两组数据）
        palette: 颜色方案
    """
    plt.figure(figsize=figsize)
    
    # 转换数据格式以适应seaborn
    if not isinstance(data, list):
        data = [data]
    
    # 准备DataFrame格式数据
    df_list = []
    for i, d in enumerate(data):
        label = labels[i] if labels else f"数据{i+1}"
        df_list.append(pd.DataFrame({x_label: label, y_label: d}))
    
    df = pd.concat(df_list, ignore_index=True)
    
    # 绘制小提琴图
    sns.violinplot(x=x_label, y=y_label, data=df, inner=inner, split=split, palette=palette)
    
    # 设置图表属性
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
        # 将图像保存到内存缓冲区
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    buffer.seek(0)  # 移动到缓冲区开头
    
    # 转换为Base64编码
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # 清理图表，避免内存泄漏
    plt.close()
    
    return ImageContent(data=image_base64, mimeType="image/png", type="image")


@mcp.tool()
def plot_contour(x, y, z, title="等高线图", x_label="X轴", y_label="Y轴",
                 figsize=(10, 8), levels=10, cmap='viridis', linewidths=0.5,
                 filled=True, colorbar=True):
    """
    绘制等高线图，用于展示三维数据在二维平面上的分布
    
    参数:
        x, y: 网格数据
        z: 对应的高度数据
        title: 图表标题
        x_label, y_label: 坐标轴标签
        figsize: 图表大小
        levels: 等高线的层级数量
        cmap: 颜色映射
        linewidths: 等高线宽度
        filled: 是否填充等高线区域
        colorbar: 是否显示颜色条
    """
    plt.figure(figsize=figsize)
    
    # 绘制等高线
    if filled:
        contourf = plt.contourf(x, y, z, levels=levels, cmap=cmap)
        plt.contour(x, y, z, levels=levels, colors='black', linewidths=linewidths)
        if colorbar:
            plt.colorbar(contourf)
    else:
        contour = plt.contour(x, y, z, levels=levels, cmap=cmap, linewidths=linewidths)
        plt.clabel(contour, inline=True, fontsize=8)
        if colorbar:
            plt.colorbar(contour)
    
    # 设置图表属性
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(linestyle='--', alpha=0.3)
    
    plt.tight_layout()
        # 将图像保存到内存缓冲区
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    buffer.seek(0)  # 移动到缓冲区开头
    
    # 转换为Base64编码
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # 清理图表，避免内存泄漏
    plt.close()
    
    return ImageContent(data=image_base64, mimeType="image/png", type="image")


@mcp.tool()
def plot_radar(data, categories, title="雷达图", figsize=(8, 8), 
               labels=None, colors=None, alpha=0.25):
    """
    绘制雷达图，用于多维度数据的对比
    
    参数:
        data: 数据，形状为(n_series, n_categories)
        categories: 每个维度的名称
        title: 图表标题
        figsize: 图表大小
        labels: 每个数据系列的标签
        colors: 每个数据系列的颜色
        alpha: 填充区域的透明度（0-1之间的数）
    """
    # 确保数据是二维数组
    if len(np.array(data).shape) == 1:
        data = [data]
    
    n_series = len(data)
    n_categories = len(categories)
    
    # 计算每个类别的角度
    angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
    
    # 闭合雷达图
    data = [np.concatenate((d, [d[0]])) for d in data]
    angles = angles + [angles[0]]
    categories = categories + [categories[0]]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # 绘制每条数据线
    for i, d in enumerate(data):
        color = colors[i] if colors else None
        ax.plot(angles, d, color=color, linewidth=2, label=labels[i] if labels else f"数据{i+1}")
        ax.fill(angles, d, color=color, alpha=float(alpha))
    
    # 设置坐标轴标签
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # 设置图表属性
    plt.title(title)
    if labels:
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    plt.tight_layout()
    return fig





@mcp.tool()
def plot_bubble(x, y, size, color=None, cmap='viridis', title="气泡图",
                x_label="X轴", y_label="Y轴", figsize=(10, 6), alpha=0.7,
                edgecolor='k', colorbar=False, s_min=10, s_max=500):
    """
    绘制气泡图，通过点的大小和颜色展示多维度数据
    
    参数:
        x, y: 基础坐标数据
        size: 气泡大小数据
        color: 气泡颜色数据（可选）
        cmap: 颜色映射
        title: 图表标题
        x_label, y_label: 坐标轴标签
        figsize: 图表大小
        alpha: 透明度（0-1之间的数）
        edgecolor: 边缘颜色
        colorbar: 是否显示颜色条
        s_min, s_max: 气泡大小的最小和最大值
    """
    plt.figure(figsize=figsize)
    
    # 标准化大小数据
    size_norm = (size - np.min(size)) / (np.max(size) - np.min(size))
    sizes = size_norm * (s_max - s_min) + s_min
    
    # 绘制气泡图
    scatter = plt.scatter(
        x, y, s=sizes, c=color, cmap=cmap,
        alpha=float(alpha), edgecolor=edgecolor
    )
    
    # 设置图表属性
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(linestyle='--', alpha=0.7)
    
    if colorbar and color is not None:
        plt.colorbar(scatter)
    
    plt.tight_layout()
        # 将图像保存到内存缓冲区
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    buffer.seek(0)  # 移动到缓冲区开头
    
    # 转换为Base64编码
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # 清理图表，避免内存泄漏
    plt.close()
    
    return ImageContent(data=image_base64, mimeType="image/png", type="image")


@mcp.tool()
def plot_ternary(x, y, z, size=None, color=None, cmap='viridis', title="三元图",
                 figsize=(10, 10), alpha=0.7, s=50, colorbar=False):
    """
    绘制三元图，用于展示三个变量之间的比例关系
    
    参数:
        x, y, z: 三个变量数据（应满足x + y + z = 1或常数）
        size: 点的大小
        color: 点的颜色
        cmap: 颜色映射
        title: 图表标题
        figsize: 图表大小
        alpha: 透明度（0-1之间的数）
        s: 点的基础大小
        colorbar: 是否显示颜色条
    """
    # 确保三个变量的和为1
    sum_xyz = x + y + z
    x = x / sum_xyz
    y = y / sum_xyz
    z = z / sum_xyz
    
    # 转换为笛卡尔坐标
    x_cart = 0.5 * (2*y + z) / (x + y + z)
    y_cart = (np.sqrt(3)/2) * z / (x + y + z)
    
    plt.figure(figsize=figsize)
    
    # 绘制三角形边界
    plt.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-')
    
    # 绘制坐标轴标签
    plt.text(1.05, 0, 'X', ha='center', va='center')
    plt.text(-0.05, 0, 'Y', ha='center', va='center')
    plt.text(0.5, np.sqrt(3)/2 + 0.05, 'Z', ha='center', va='center')
    
    # 绘制数据点
    scatter = plt.scatter(
        x_cart, y_cart, c=color, s=size if size is not None else s,
        cmap=cmap, alpha=float(alpha), edgecolors='k'
    )
    
    # 设置图表属性
    plt.title(title)
    plt.axis('equal')
    plt.axis('off')
    
    if color is not None and colorbar:
        plt.colorbar(scatter)
    
    plt.tight_layout()
        # 将图像保存到内存缓冲区
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    buffer.seek(0)  # 移动到缓冲区开头
    
    # 转换为Base64编码
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # 清理图表，避免内存泄漏
    plt.close()
    
    return ImageContent(data=image_base64, mimeType="image/png", type="image")


@mcp.tool()
def plot_correlation_matrix(data, title="相关性矩阵热图", figsize=(12, 10),
                            cmap="coolwarm", annot=True, fmt=".2f", method='pearson'):
    """
    绘制相关性矩阵热图
    
    参数:
        data: DataFrame，包含多个变量
        title: 图表标题
        figsize: 图表大小
        cmap: 颜色映射
        annot: 是否显示相关系数值
        fmt: 数值显示格式
        method: 相关系数计算方法，可选'pearson', 'kendall', 'spearman'
    """
    plt.figure(figsize=figsize)
    
    # 计算相关系数
    corr = data.corr(method=method)
    
    # 绘制热力图
    mask = np.triu(np.ones_like(corr, dtype=bool))  # 仅显示下三角
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=annot, fmt=fmt,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    # 设置图表属性
    plt.title(title)
    
    plt.tight_layout()
        # 将图像保存到内存缓冲区
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    buffer.seek(0)  # 移动到缓冲区开头
    
    # 转换为Base64编码
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # 清理图表，避免内存泄漏
    plt.close()
    
    return ImageContent(data=image_base64, mimeType="image/png", type="image")

    



# @mcp.tool()
# def predict_superconductor(
#     material_features: Dict[str, Union[float, int]],
#     model_path: str = "superconductor_mixed_model.h5",
#     scaler_path: str = "scaler_mixed.pkl",
#     prob_threshold: float = 0.5
# ) -> Dict[str, Union[str, float, bool]]:
#     """
#     预测单个物质是否为超导体（概率）及临界温度的工具函数
    
#     参数说明：
#         material_features: 单个物质的特征字典，需包含训练时所有特征字段（number_of_elements	mean_atomic_mass	wtd_mean_atomic_mass	gmean_atomic_mass	wtd_gmean_atomic_mass	entropy_atomic_mass	wtd_entropy_atomic_mass	range_atomic_mass	wtd_range_atomic_mass	
#         std_atomic_mass	wtd_std_atomic_mass	mean_fie	wtd_mean_fie	gmean_fie	wtd_gmean_fie	entropy_fie	wtd_entropy_fie	range_fie	wtd_range_fie	std_fie	wtd_std_fie	mean_atomic_radius	wtd_mean_atomic_radius	gmean_atomic_radius	wtd_gmean_atomic_radius	
#         entropy_atomic_radius	wtd_entropy_atomic_radius	range_atomic_radius	wtd_range_atomic_radius	std_atomic_radius	wtd_std_atomic_radius	mean_Density	wtd_mean_Density	gmean_Density	wtd_gmean_Density	entropy_Density	wtd_entropy_Density	range_Density	
#         wtd_range_Density	std_Density	wtd_std_Density	mean_ElectronAffinity	wtd_mean_ElectronAffinity	gmean_ElectronAffinity	wtd_gmean_ElectronAffinity	entropy_ElectronAffinity	wtd_entropy_ElectronAffinity	range_ElectronAffinity	wtd_range_ElectronAffinity	
#         std_ElectronAffinity	wtd_std_ElectronAffinity	mean_FusionHeat	wtd_mean_FusionHeat	gmean_FusionHeat	wtd_gmean_FusionHeat	entropy_FusionHeat	wtd_entropy_FusionHeat	range_FusionHeat	wtd_range_FusionHeat	std_FusionHeat	wtd_std_FusionHeat	
#         mean_ThermalConductivity	wtd_mean_ThermalConductivity	gmean_ThermalConductivity	wtd_gmean_ThermalConductivity	entropy_ThermalConductivity	wtd_entropy_ThermalConductivity	range_ThermalConductivity	wtd_range_ThermalConductivity	
#         std_ThermalConductivity	wtd_std_ThermalConductivity	mean_Valence	wtd_mean_Valence	gmean_Valence	wtd_gmean_Valence	entropy_Valence	wtd_entropy_Valence	range_Valence	wtd_range_Valence	std_Valence	wtd_std_Valence	critical_temp
# ），
#                           注意：**不包含critical_temp字段**（该字段为训练标签，预测时无需输入）
#         model_path: 训练好的混合模型路径（默认：superconductor_mixed_model.h5）
#         scaler_path: 训练时保存的标准化器路径（默认：scaler_mixed.pkl）
#         prob_threshold: 超导判断阈值（默认0.5，概率>阈值判定为"可能超导"）
    
#     返回结果：
#         包含预测结论、超导概率、预测临界温度、判断依据的字典
#     """
    
#     try:
#         if GLOBAL_MODEL is None:
#             GLOBAL_MODEL = load_model(model_path)
#         if GLOBAL_SCALER is None:
#             GLOBAL_SCALER = joblib.load(scaler_path)
#     except Exception as e:
#         return {
#             "预测状态": "失败",
#             "错误信息": f"模型/标准化器加载失败：{str(e)}",
#             "建议": "检查模型文件路径、完整性及TensorFlow版本兼容性"
#         }
    
#     try:
#         required_features = GLOBAL_SCALER.feature_names_in_  # 标准化器保存的训练集特征名
#     except AttributeError:
#         return {
#             "预测状态": "失败",
#             "错误信息": "标准化器未保存训练集特征列表（可能是旧版本生成的scaler）",
#             "建议": "重新用新版本sklearn训练scaler并保存"
#         }
    
#     # 步骤2：检查输入特征是否完整（无缺失、无多余）
#     input_features = set(material_features.keys())
#     required_set = set(required_features)
    
#     # 缺失特征检查
#     missing_features = required_set - input_features
#     if missing_features:
#         return {
#             "预测状态": "失败",
#             "错误信息": f"输入特征缺失：{sorted(list(missing_features))}",
#             "建议": f"需补充以下必填特征：{sorted(list(required_features))}"
#         }
    
#     # 多余特征检查（避免干扰预测）
#     extra_features = input_features - required_set
#     if extra_features:
#         # 仅警告，不阻断预测（自动过滤多余特征）
#         print(f"⚠️  输入包含多余特征，将自动过滤：{sorted(list(extra_features))}")
    

#     material_df = pd.DataFrame([material_features])[required_features]
    
#     # 步骤2：特征标准化（使用训练集拟合的scaler，避免数据泄露）
#     try:
#         material_scaled = GLOBAL_SCALER.transform(material_df)
#     except Exception as e:
#         return {
#             "预测状态": "失败",
#             "错误信息": f"特征标准化失败：{str(e)}",
#             "建议": "检查输入特征值是否为有效数字（如无字符串、无穷大等）"
#         }
    
#     try:
#         # 模型输出：[超导概率预测值, 临界温度预测值]
#         pred_prob, pred_temp = GLOBAL_MODEL.predict(material_scaled, verbose=0)
        
#         # 结果格式化（保留4位小数，提升可读性）
#         superconductor_prob = round(float(pred_prob[0][0]), 4)  # 超导概率（0-1）
#         predicted_temp = round(float(pred_temp[0][0]), 4)       # 预测临界温度（K）
#         is_superconductor = superconductor_prob > prob_threshold  # 是否判定为超导
        
#         # 温度预测合理性修正（非超导物质温度理论上应≥0，避免负温度输出）
#         predicted_temp = max(predicted_temp, 0.0)
    
#     except Exception as e:
#         return {
#             "预测状态": "失败",
#             "错误信息": f"模型预测过程出错：{str(e)}",
#             "建议": "检查输入特征值范围是否与训练集一致（如无异常极大/极小值）"
#         }
    
#     conclusion = "该物质**可能是超导体**" if is_superconductor else "该物质**大概率非超导体**"
#     basis = f"超导概率（{superconductor_prob}）{'超过' if is_superconductor else '未超过'}判定阈值（{prob_threshold}）"
    
#     return {
#         "预测状态": "成功",
#         "物质超导结论": conclusion,
#         "判断依据": basis,
#         "超导概率（0-1）": superconductor_prob,
#         "预测临界温度（K）": predicted_temp,
#         "补充说明": [
#             "1. 临界温度仅为预测值，实际需实验验证",
#             "2. 非超导物质的预测温度接近0K为正常结果",
#             "3. 可调整prob_threshold参数优化判断灵敏度（默认0.5）"
#         ]
#     }

import re
import numpy as np

# Pauling电负性数据（扩展常用超导相关元素）
_pauling_en = {
    "H": 2.20, "He": 0, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55,
    "N": 3.04, "O": 3.44, "F": 3.98, "Na": 0.93, "Mg": 1.31, "Al": 1.61,
    "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16, "K": 0.82, "Ca": 1.00,
    "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88,
    "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18,
    "Se": 2.55, "Br": 2.96, "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33,
    "Nb": 1.60, "Mo": 2.16, "Tc": 1.90, "Ru": 2.20, "Rh": 2.28, "Pd": 2.20,
    "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05, "Te": 2.10,
    "I": 2.66, "Cs": 0.79, "Ba": 0.89, "La": 1.10, "Ce": 1.12, "Pr": 1.13,
    "Nd": 1.14, "Sm": 1.17, "Eu": 1.20, "Gd": 1.20, "Tb": 1.24, "Dy": 1.22,
    "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.10, "Lu": 1.27, "Hf": 1.30,
    "Ta": 1.50, "W": 2.36, "Re": 1.90, "Os": 2.20, "Ir": 2.20, "Pt": 2.28,
    "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 2.33, "Bi": 2.02
}

# 元素原子序数映射表
_atomic_numbers = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
    "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32,
    "As": 33, "Se": 34, "Br": 35, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47,
    "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Cs": 55,
    "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Sm": 62, "Eu": 63,
    "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77,
    "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83
}



@mcp.tool()
def predict_superconductor(mp_entry):
    """
    在Materials Project中调用的超导特性预测函数
    
    参数:
        mp_entry: Materials Project条目字典，需包含以下字段:
            - composition_reduced: 简化化学式
            - formation_energy_per_atom: 形成能 (eV/atom)
            - efermi: 费米能级 (eV)
            - volume: 晶胞体积 (Å³)
            - nsites: 晶胞中的原子数
            - is_metal: 是否为金属 (布尔值)
            - total_magnetization: 总磁矩 (μB)
            - bulk_modulus: 体弹性模量 (GPa)
            - shear_modulus: 剪切模量 (GPa)
            - density: 密度 (g/cm³)
            - average_atomic_mass: 平均原子质量 (amu)
    
    示例输入：
        {
            "material_id": "mp-1234",
            "composition_reduced": "NbTi",
            "formation_energy_per_atom": -0.5,
            "efermi": 5.0,
            "volume": 50.0,
            "nsites": 2,
            "is_metal": True,
            "total_magnetization": 0.0,
            "bulk_modulus": 160.0,
            "shear_modulus": 70.0,
            "density": 6.5,
            "average_atomic_mass": 47.867
        }
    
    
    返回:
        dict: 包含超导特性预测结果的字典
    """
    # 初始化结果字典
    result = {
        "material_id": mp_entry.get("material_id", "unknown"),
        "composition": mp_entry["composition_reduced"],
        "prediction": {
            "critical_temperature_k": 0.0,
            "formation_energy_ev_per_atom": 0.0,
            "energy_gap_ev": 0.0,
            "is_potential_superconductor": False
        },
        "parameters": {},
        "status": "processed",
        "message": ""
    }
    
    # 1. 初步筛选：必须是金属且低磁矩
    if not mp_entry.get("is_metal", False):
        result["status"] = "filtered"
        result["message"] = "非金属材料，排除超导可能性"
        return result
        
    if abs(mp_entry.get("total_magnetization", 1.0)) > 0.1:
        result["status"] = "filtered"
        result["message"] = "磁矩过高，排除超导可能性"
        return result
    
    try:
        # 2. 解析化学式并计算平均原子序数
        parsed_comp = _parse_composition(mp_entry["composition_reduced"])
        avg_atomic_num = _calculate_average_atomic_number(parsed_comp)
        
        # 3. 计算体积/原子 (转换为m³)
        volume_per_atom = (mp_entry["volume"] * 1e-30) / mp_entry["nsites"]
        
        # 4. 计算电负性差异
        en_diff = _calculate_electronegativity_diff(parsed_comp)
        
        # 5. 估算德拜温度
        debye_temp = _estimate_debye_temperature(
            bulk_modulus=mp_entry["bulk_modulus"],
            shear_modulus=mp_entry["shear_modulus"],
            density=mp_entry["density"],
            volume_per_atom=volume_per_atom
        )
        
        # 6. 估算声子频率 (THz)
        phonon_freq = _estimate_phonon_frequency(debye_temp)
        
        # 7. 估算电子-声子耦合强度
        electron_phonon_coupling = _estimate_electron_phonon_coupling(
            bulk_modulus=mp_entry["bulk_modulus"],
            shear_modulus=mp_entry["shear_modulus"]
        )
        
        # 8. 计算临界温度Tc (K)
        critical_temp = _calculate_critical_temperature(
            debye_temperature=debye_temp,
            electron_phonon_coupling=electron_phonon_coupling
        )
        
        # 9. 计算能隙 (eV)
        energy_gap = _calculate_energy_gap(
            critical_temperature=critical_temp,
            fermi_energy=mp_entry["efermi"]
        )
        
        # 10. 提取形成能
        formation_energy = mp_entry["formation_energy_per_atom"]
        
        # 11. 判断是否为潜在超导体
        is_superconductor = (critical_temp > 5) and (formation_energy < 0) and (energy_gap < 0.05)
        
        # 更新结果
        result["prediction"] = {
            "critical_temperature_k": round(critical_temp, 2),
            "formation_energy_ev_per_atom": round(formation_energy, 3),
            "energy_gap_ev": round(energy_gap, 4),
            "is_potential_superconductor": is_superconductor
        }
        
        # 保存中间参数
        result["parameters"] = {
            "average_atomic_number": round(avg_atomic_num, 2),
            "volume_per_atom_m3": volume_per_atom,
            "electronegativity_difference": round(en_diff, 2),
            "debye_temperature_k": round(debye_temp, 2),
            "phonon_frequency_thz": round(phonon_freq, 2),
            "electron_phonon_coupling": round(electron_phonon_coupling, 3)
        }
        
        result["message"] = "预测完成"
        
    except Exception as e:
        result["status"] = "error"
        result["message"] = f"计算过程出错: {str(e)}"
    
    return result


@mcp.tool()
def _parse_composition(composition_str):
    """解析化学式字符串为元素-比例字典"""
    elements = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", composition_str)
    parsed = {}
    for elem, ratio in elements:
        parsed[elem] = float(ratio) if ratio else 1.0
    total = sum(parsed.values())
    return {k: v/total for k, v in parsed.items()}

@mcp.tool()
def _calculate_average_atomic_number(parsed_composition):
    """计算平均原子序数"""
    avg = 0.0
    for elem, ratio in parsed_composition.items():
        avg += _atomic_numbers.get(elem, 0) * ratio
    return avg

@mcp.tool()
def _calculate_electronegativity_diff(parsed_composition):
    """计算元素间最大电负性差异"""
    ens = [_pauling_en.get(elem, 0) for elem in parsed_composition]
    if len(ens) < 2:
        return 0.0
    return max(ens) - min(ens)

@mcp.tool()
def _estimate_debye_temperature(bulk_modulus, shear_modulus, density, volume_per_atom):
    """使用弹性模量估算德拜温度"""
    if shear_modulus <= 0 or density <= 0:
        return 100.0  # 默认值
    
    # 转换单位：GPa -> Pa，g/cm³ -> kg/m³
    shear_modulus_pa = shear_modulus * 1e9
    density_kgm3 = density * 1000
    
    # 估算声速
    sound_velocity = np.sqrt(shear_modulus_pa / density_kgm3)
    
    # 原子数密度 (1/m³)
    atom_density = 1 / volume_per_atom
    
    # 德拜温度计算
    theta_d = (6.626e-34 / (2 * np.pi * 1.38e-23)) * \
              (3 * atom_density / (4 * np.pi))**(1/3) * \
              sound_velocity
    
    return max(theta_d, 10.0)  # 确保合理的最低值

@mcp.tool()
def _estimate_phonon_frequency(debye_temperature):
    """从德拜温度估算声子频率 (THz)"""
    # 德拜频率与德拜温度关系：ω_D = (k_B * θ_D) / ħ
    # 转换为THz (1 THz = 1e12 Hz)
    return (1.38e-23 * debye_temperature) / (1.054e-34 * 1e12)

@mcp.tool()
def _estimate_electron_phonon_coupling(bulk_modulus, shear_modulus):
    """估算电子-声子耦合强度"""
    if shear_modulus <= 0:
        return 0.2  # 默认弱耦合
    
    # 基于弹性模量比的经验公式
    ratio = bulk_modulus / shear_modulus
    lambda_ = 0.05 + 0.4 * (ratio - 1.5)
    
    # 限制在物理合理范围内
    return max(0.1, min(lambda_, 2.0))

@mcp.tool()
def _calculate_critical_temperature(debye_temperature, electron_phonon_coupling):
    """使用简化的McMillan公式计算临界温度"""
    mu_star = 0.1  # 有效库仑排斥参数
    
    if electron_phonon_coupling <= mu_star:
        return 0.0
    
    return debye_temperature * np.exp(-1 / (electron_phonon_coupling - mu_star))

@mcp.tool()
def _calculate_energy_gap(critical_temperature, fermi_energy):
    """计算超导能隙"""
    if critical_temperature > 0:
        # BCS理论：E_gap ≈ 3.52 * k_B * T_c (转换为eV)
        return 3.52 * 8.617e-5 * critical_temperature
    else:
        # 非超导状态下基于费米能级的近似
        return 0.01 * abs(fermi_energy)
    

@mcp.tool()
def search_arXiv(query: str, max_results: int) -> TextContent:
    """Search papers from the online database arxiv.org

    Args:
        query: the content(key words, concepts, etc) to be searched on arXiv
        max_results: the maxiumum number of result papers to be returned

    Return: the basic information of the result papers searched on arXiv,
            including id, primary category, published time, pdf url link, abstract, comments, abstract url link"""

    rst = ''
    def output(content):
        nonlocal rst
        rst += content
        rst += '\n'

    url = f'http://export.arxiv.org/api/query?search_query={query}&start=0&max_results={max_results}'
    response = requests.get(url)
    xml_response = response.text  # string
    # print(xml_response)

    root = ET.fromstring(xml_response)  # 从字符串解析

    # 命名空间
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'opensearch': 'http://a9.com/-/spec/opensearch/1.1/',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }

    total_results = root.find('opensearch:totalResults', ns).text
    start_index = root.find('opensearch:startIndex', ns).text
    items_per_page = root.find('opensearch:itemsPerPage', ns).text

    output(f"找到 {total_results} 篇符合条件的论文。")
    output(f"当前显示第 {start_index} 到 {int(start_index) + int(items_per_page) - 1} 篇。\n")

    def read_entry(entry):
        if entry is not None:
            paper_id = entry.find('atom:id', ns).text
            title = entry.find('atom:title', ns).text.strip()  # .strip() 去除多余空白符
            published = entry.find('atom:published', ns).text
            summary = entry.find('atom:summary', ns).text.strip()
            author_name = entry.find('atom:author/atom:name', ns).text  # 使用XPath语法查找

            comment_elem = entry.find('arxiv:comment', ns)
            journal_ref_elem = entry.find('arxiv:journal_ref', ns)
            comment = comment_elem.text if comment_elem is not None else "N/A"
            journal_ref = journal_ref_elem.text if journal_ref_elem is not None else "N/A"

            primary_category = entry.find('arxiv:primary_category', ns).get('term')

            for link in entry.findall('atom:link', ns):

                if link.get('title') == 'pdf':
                    pdf_url = link.get('href')

                if link.get('rel') == 'alternate':
                    abs_url = link.get('href')

            output("【论文信息】")
            output(f"标题: {title}")
            output(f"ID: {paper_id}")
            output(f"发布时间: {published}")
            output(f"作者: {author_name}")
            output(f"主分类: {primary_category}")
            output(f"摘要: {summary[:200]}...")  # 打印前200个字符以避免刷屏
            output(f"评论: {comment}")
            output(f"期刊引用: {journal_ref}")
            output(f"摘要页面: {abs_url}")
            output(f"PDF链接: {pdf_url}")
            output("\n <--------------------------------------------> \n")

        else:
            output("未找到任何论文。")

    # entry = root.find('atom:entry', ns)
    all_entries = root.findall('atom:entry', ns)

    cnt = 0
    for entry in all_entries:
        output(f"Entry #{cnt}")
        read_entry(entry)
        cnt += 1

    return TextContent(type="text", text = rst)

@mcp.tool()
async def read_pdf_tool(url: str) -> CallToolResult:
    """
    读取给定URL的PDF文件内容并返回给客户端

    此工具专门处理PDF文件，如果不是PDF格式则返回错误。

    参数:
        url: 待读取的pdf文件的url

    返回:
        符合MCP协议的CallToolResult，包含PDF内容
    """
    content_blocks: List[ContentBlock] = []
    is_error = False

    try:
        # 验证URL格式
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            error_text = TextContent(
                type="text",
                text=f"URL格式不正确: {url}",
                annotations=Annotations(audience=["user"])
            )
            content_blocks.append(error_text)
            return CallToolResult(content=content_blocks, isError=True)

        # 发送HTTP请求获取内容
        headers = {
            'User-Agent': 'MCP-PDF-Reader/1.0'
        }

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        # 检查内容类型是否为PDF
        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' not in content_type:
            # 如果不是PDF，返回错误
            error_text = TextContent(
                type="text",
                text=f"URL指向的内容不是PDF格式: {content_type}",
                annotations=Annotations(audience=["user"])
            )
            content_blocks.append(error_text)
            return CallToolResult(content=content_blocks, isError=True)

        # 获取PDF内容
        pdf_content = response.content
        content_length = len(pdf_content)

        # 创建二进制内容对象
        blob_content = BlobResourceContents(
            uri=url,
            mimeType="application/pdf",
            blob=base64.b64encode(pdf_content).decode('utf-8'),
            size=content_length
        )

        # 创建嵌入式资源
        embedded_pdf = EmbeddedResource(
            type="resource",
            resource=blob_content,
            annotations=Annotations(
                audience=["assistant"],
                priority=0.8
            )
        )
        content_blocks.append(embedded_pdf)

        # 添加成功信息
        success_text = TextContent(
            type="text",
            text=f"成功获取PDF文件: {url}\n大小: {content_length} 字节",
            annotations=Annotations(audience=["user"])
        )
        content_blocks.insert(0, success_text)

    except requests.exceptions.RequestException as e:
        # 处理网络请求错误
        error_text = TextContent(
            type="text",
            text=f"获取PDF文件时出错: {str(e)}",
            annotations=Annotations(audience=["user"])
        )
        content_blocks.append(error_text)
        is_error = True

    except Exception as e:
        # 处理其他错误
        error_text = TextContent(
            type="text",
            text=f"处理PDF文件时出错: {str(e)}",
            annotations=Annotations(audience=["user"])
        )
        content_blocks.append(error_text)
        is_error = True

    # 返回结果
    return CallToolResult(
        content=content_blocks,
        isError=is_error
    )
computing_tools.register(mcp)

# --- Materials Project 工具 ---
@mcp.tool()
def search_materials_by_formula(chemical_formula: str) -> TextContent:
    """
    根据化学式在 Materials Project 数据库中搜索材料。
    :param chemical_formula: 要搜索的化学式 (例如: "Fe2O3", "SiC")。
    :return: 包含材料ID和化学式的文本结果。
    """
    try:
        with MPRester(API_KEY) as mpr:
            # all_fields=True 已被弃用, fields=["material_id", "formula_pretty"] 是推荐用法
            results = mpr.materials.search(
                formula=chemical_formula,
                fields=["material_id", "formula_pretty"]
            )
            if not results:
                return TextContent(type="text", text="未找到任何结果。")
            descs = []
            for m in results:
                mat_id = getattr(m, "material_id", "N/A")
                formula = getattr(m, "formula_pretty", "N/A")
                descs.append(f"{mat_id}: {formula}")
            return TextContent(type="text", text="\n".join(descs))
    except Exception as e:
        logger.error(f"搜索材料时发生错误: {e}", exc_info=True)
        return TextContent(type="text", text=f"错误: {e}")


def safe_get(obj, path, default="N/A"):
    """安全获取嵌套属性，避免 KeyError/AttributeError。"""
    try:
        for p in path.split("."):
            obj = getattr(obj, p)
        return obj if obj is not None else default
    except Exception:
        return default

@mcp.tool()
def search_info(material_ids: str) -> TextContent | BlobResourceContents:
    """
    根据 Materials Project material_id 查询详细信息。
    输入可以是单个 ID 或多个 ID（用逗号隔开）。
    - 单个 ID: 返回详细信息（文本形式）
    - 多个 ID: 返回表格文本 + 导出 CSV 文件
    """
    try:
        ids = [mid.strip() for mid in material_ids.split(",") if mid.strip()]
        if not ids:
            return TextContent(type="text", text="请输入至少一个材料编号。")

        with MPRester(API_KEY) as mpr:
            results = mpr.summary.search(
                material_ids=ids,
                fields=[
                    "material_id",
                    "formula_pretty",
                    "symmetry.spacegroup.symbol",
                    "band_gap",
                    "density",
                    "formation_energy_per_atom",
                ],
            )

        if not results:
            return TextContent(type="text", text="未找到任何结果。")

        # 单个 ID → 返回详细信息
        if len(ids) == 1:
            m = results[0]
            info = (
                f"材料编号: {safe_get(m, 'material_id')}\n"
                f"化学式: {safe_get(m, 'formula_pretty')}\n"
                f"空间群: {safe_get(m, 'symmetry.spacegroup.symbol')}\n"
                f"带隙 (eV): {safe_get(m, 'band_gap')}\n"
                f"密度 (g/cm³): {safe_get(m, 'density')}\n"
                f"生成能量 (eV/atom): {safe_get(m, 'formation_energy_per_atom')}\n"
            )
            return TextContent(type="text", text=info)

        # 多个 ID → 返回表格 & 文件
        else:
            header = "material_id,formula,spacegroup,band_gap(eV),density(g/cm3),E_form(eV/atom)"
            lines = [header]
            for m in results:
                lines.append(
                    f"{safe_get(m, 'material_id')},"
                    f"{safe_get(m, 'formula_pretty')},"
                    f"{safe_get(m, 'symmetry.spacegroup.symbol')},"
                    f"{safe_get(m, 'band_gap')},"
                    f"{safe_get(m, 'density')},"
                    f"{safe_get(m, 'formation_energy_per_atom')}"
                )
            csv_content = "\n".join(lines)

            buffer = io.BytesIO(csv_content.encode("utf-8"))
            return BlobResourceContents(
                blob=buffer.getvalue(),
                mime_type="text/csv",
                suggested_filename="materials_info.csv"
            )

    except Exception as e:
        logger.error(f"查询材料信息时出错: {e}", exc_info=True)
        return TextContent(type="text", text=f"错误: {e}")
# --- 新增的凝聚态计算工具 (使用 ASE) ---
@mcp.tool()
def calculate_total_energy_ase(
    chemical_symbol: str,
    lattice_constant: float,
    crystal_structure: str = "fcc"
) -> TextContent:
    """
    使用 ASE 和 EMT 计算器计算晶体结构的总能量。
    这是一个简化的模型，适用于金属，不适用于复杂的化合物。

    :param chemical_symbol: 元素的化学符号 (例如: "Cu", "Al", "Pt")。
    :param lattice_constant: 晶格常数 a (单位: 埃, Å)。
    :param crystal_structure: 晶体结构类型。支持 "fcc", "bcc", "hcp", "diamond" 等ASE支持的结构。默认为 "fcc"。
    :return: 计算出的总能量或错误信息。
    """
    logger.info(
        f"收到能量计算请求: 元素={chemical_symbol}, "
        f"晶格常数={lattice_constant}, 结构={crystal_structure}"
    )
    try:
        # 1. 使用 ase.build.bulk 创建原子结构
        #    'bulk' 函数可以方便地构建常见晶体结构
        atoms = bulk(name=chemical_symbol, crystalstructure=crystal_structure, a=lattice_constant, cubic=True)
        
        # 2. 设置计算器
        #    EMT (Effective Medium Theory) 是一个非常快速的近似计算器，适合演示
        atoms.calc = EMT()

        # 3. 运行计算并获取总能量
        #    get_potential_energy() 会触发实际的计算过程
        total_energy = atoms.get_potential_energy()

        # 4. 格式化并返回结果
        result_text = (
            f"计算成功。\n"
            f"元素: {chemical_symbol}\n"
            f"晶体结构: {crystal_structure}\n"
            f"晶格常数: {lattice_constant} Å\n"
            f"---------------------------\n"
            f"总能量: {total_energy:.4f} eV"
        )
        return TextContent(type="text", text=result_text)

    except Exception as e:
        logger.error(f"ASE 计算失败: {e}", exc_info=True)
        return TextContent(type="text", text=f"计算失败: {e}. 请检查输入的化学符号或晶体结构是否有效。")

@mcp.tool()
def search_in_SuperCon(formulas: str) -> TextContent:
    """
    根据化学式在 SuperCon 数据集中搜索超导体信息。
    formulas: 单个或多个化学式, 用逗号分隔
    """
    if df_supercon.empty:
        return TextContent(type="text", text="SuperCon dataset not loaded.")

    formulas_list = [f.strip() for f in formulas.split(",") if f.strip()]
    results_texts = []

    for formula in formulas_list:
        subset = df_supercon[df_supercon['formula'].str.replace(" ", "").str.lower() == formula.replace(" ", "").lower()]
        if subset.empty:
            results_texts.append(f"{formula}: No entry found in SuperCon dataset.")
            continue

        # 遍历匹配的材料
        for _, row in subset.iterrows():
            info = f"""
Material: {row.get('rawMaterial', 'N/A')}
Formula: {row.get('formula', 'N/A')}
Doping: {row.get('doping', 'N/A')}
Shape: {row.get('shape', 'N/A')}
Material class: {row.get('materialClass', 'N/A')}
Fabrication: {row.get('fabrication', 'N/A')}
Substrate: {row.get('substrate', 'N/A')}
Critical Temperature: {row.get('criticalTemperature', 'N/A')} K
Measurement Method: {row.get('criticalTemperatureMeasurementMethod', 'N/A')}
Applied Pressure: {row.get('appliedPressure', 'N/A')}
Reference: {row.get('doi', 'N/A')}
Authors: {row.get('authors', 'N/A')}
Journal: {row.get('journal', 'N/A')}, Year: {row.get('year', 'N/A')}
"""
            results_texts.append(info.strip())

    return TextContent(type="text", text="\n\n".join(results_texts))


# This is the main entry point for your server
def main():
    logger.info('Starting your-new-server')
    mcp.run('stdio')


if __name__ == "__main__":
    main()

def main():
    logger.info("Materials Project Full Visualizer MCP Server starting...")
    logger.info(f"API Key Configured: {'Yes' if API_KEY != 'your_materials_project_api_key_here' else 'No (Please set MP_API_KEY environment variable)'}")
    mcp.run('stdio')  # 标准IO模式运行，适配MCP框架

if __name__ == "__main__":
    main()