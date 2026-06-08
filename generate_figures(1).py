"""
生成报告所需的补充图表：
1. 软件流程图
2. 循环结构示意图
3. 三种映射轨迹图
4. 安全性对比雷达图
5. 置换矩阵热力图（3种映射对比）
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrow
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

_font_path = "/System/Library/Fonts/Supplemental/Songti.ttc"
fm.fontManager.addfont(_font_path)
_prop = fm.FontProperties(fname=_font_path)
plt.rcParams["font.family"] = _prop.get_name()
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/kai/Desktop/practice/2026/05/30__SquidwardCoIfan/"


# 1. 软件流程图
def draw_flowchart():
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 18)
    ax.axis("off")

    def box(x, y, w, h, text, color="#4A90D9", text_color="white", style="round,pad=0.1"):
        bbox = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle=style,
                              facecolor=color, edgecolor="white",
                              linewidth=1.5, zorder=3)
        ax.add_patch(bbox)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=10, color=text_color, zorder=4,
                wrap=True)

    def diamond(x, y, w, h, text, color="#E67E22"):
        xs = [x, x+w/2, x, x-w/2, x]
        ys = [y+h/2, y, y-h/2, y, y+h/2]
        ax.fill(xs, ys, color=color, zorder=3)
        ax.plot(xs, ys, color="white", linewidth=1.5, zorder=4)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=9.5, color="white", zorder=5)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#555555",
                                   lw=1.8), zorder=2)

    # 节点坐标（从上到下）
    steps = [
        (5, 17,   4, 0.9, "开始", "#2ECC71"),
        (5, 15.5, 4, 0.9, "输入：映射类型、参数 μ/k\n初值 x₀、置换长度 N", "#4A90D9"),
        (5, 13.8, 4, 0.9, "预热迭代 M=1000 步\n消除初始影响", "#4A90D9"),
        (5, 12.1, 4, 0.9, "继续迭代 N 步\n得到混沌序列 {xᵢ}", "#4A90D9"),
        (5, 10.4, 4, 0.9, "argsort 排序\n生成置换 σ", "#4A90D9"),
        (5,  8.7, 4, 0.9, "深度优先遍历\n分解为不相交循环", "#4A90D9"),
        (5,  7.0, 4, 0.9, "计算各循环长度 ℓ₁, ℓ₂, …, ℓₖ", "#4A90D9"),
        (5,  5.3, 4, 0.9, "计算 lcm(ℓ₁, ℓ₂, …, ℓₖ)\n得到置换的阶", "#4A90D9"),
        (5,  3.6, 4, 0.9, "输出：循环分布、置换阶\n绘制统计图表", "#9B59B6"),
        (5,  2.0, 4, 0.9, "结束", "#E74C3C"),
    ]

    for (x, y, w, h, text, color) in steps:
        box(x, y, w, h, text, color=color)

    for i in range(len(steps) - 1):
        _, y1, _, h1, _, _ = steps[i]
        _, y2, _, h2, _, _ = steps[i+1]
        arrow(5, y1 - h1/2, 5, y2 + h2/2)

    ax.set_title("混沌置乱表生成与循环阶分析流程", fontsize=13, pad=10, fontweight="bold")
    plt.tight_layout()
    path = OUT + "flowchart.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {path}")


# 2. 循环结构示意图
def draw_cycle_demo():
    """用一个简单的6元素置换演示循环结构"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    perm = [1, 2, 0, 4, 3, 5]
    cycles = [[0, 1, 2], [3, 4], [5]]
    colors = ["#E74C3C", "#3498DB", "#2ECC71"]
    labels = ["长度3循环", "长度2循环", "不动点（长度1）"]

    # 左图：置换箭头图
    ax = axes[0]
    ax.set_xlim(-0.8, 5.8)
    ax.set_ylim(-2.2, 1.8)
    ax.axis("off")
    ax.set_title("置换的映射关系", fontsize=13, fontweight="bold", pad=10)

    # 画元素节点
    for i in range(6):
        c = colors[[i in cyc for cyc in cycles].index(True)]
        circle = plt.Circle((i, 0), 0.38, color=c, zorder=3)
        ax.add_patch(circle)
        ax.text(i, 0, str(i), ha="center", va="center",
                fontsize=13, color="white", fontweight="bold", zorder=4)

    # 画箭头
    for i, j in enumerate(perm):
        if i == j:
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(i + 0.38*np.cos(theta)*0.7,
                    0.38*np.sin(theta)*2.0 + 0.8,
                    color=colors[2], linewidth=2.0)
            ax.annotate("", xy=(i + 0.12, 0.38), xytext=(i - 0.12, 0.38),
                        arrowprops=dict(arrowstyle="->", color=colors[2], lw=1.8))
        else:
            c = colors[[i in cyc for cyc in cycles].index(True)]
            rad = 0.3 if abs(i-j) > 1 else 0.0
            ax.annotate("", xy=(j, 0), xytext=(i, 0),
                        arrowprops=dict(arrowstyle="->", color=c, lw=2,
                                        connectionstyle=f"arc3,rad={-rad}"))

    # σ表达式
    ax.text(2.5, -1.2,
            "σ = [1,2,0,4,3,5]",
            ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="#ECF0F1", edgecolor="#BDC3C7"))
    ax.text(2.5, -1.8,
            "ord(σ) = lcm(3, 2, 1) = 6",
            ha="center", va="center", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#FDEBD0", edgecolor="#E59866"))

    legend_handles = [mpatches.Patch(color=c, label=l)
                      for c, l in zip(colors, labels)]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.02), fontsize=10, ncol=3,
              framealpha=0.9, edgecolor="#BDC3C7")

    # 右图：循环圆圈图
    ax2 = axes[1]
    ax2.set_xlim(-3.2, 3.2)
    ax2.set_ylim(-2.5, 2.2)
    ax2.axis("off")
    ax2.set_title("循环分解示意", fontsize=13, fontweight="bold", pad=10)

    centers = [(-1.8, 0.3), (1.0, 0.5), (2.4, 0.3)]
    radii   = [0.80, 0.60, 0.32]
    cycle_labels = [["0", "1", "2"], ["3", "4"], ["5"]]
    cycle_titles = ["3-循环\n(0→1→2→0)", "2-循环\n(3→4→3)", "不动点\n(5→5)"]

    for idx, (cx, cy) in enumerate(centers):
        r = radii[idx]
        nodes = cycle_labels[idx]
        n = len(nodes)
        node_colors = colors[idx]

        angles = [np.pi/2 + 2*np.pi*k/n for k in range(n)]
        xs = [cx + r * np.cos(a) for a in angles]
        ys = [cy + r * np.sin(a) for a in angles]

        for k in range(n):
            nx, ny = xs[(k+1) % n], ys[(k+1) % n]
            ax2.annotate("", xy=(nx, ny), xytext=(xs[k], ys[k]),
                         arrowprops=dict(arrowstyle="->", color=node_colors,
                                         lw=2, connectionstyle="arc3,rad=0.3"))

        for k, (x, y) in enumerate(zip(xs, ys)):
            circle = plt.Circle((x, y), 0.24, color=node_colors, zorder=3)
            ax2.add_patch(circle)
            ax2.text(x, y, nodes[k], ha="center", va="center",
                     fontsize=12, color="white", fontweight="bold", zorder=4)

        # 标签放在循环正下方，留足空间
        ax2.text(cx, cy - r - 0.42, cycle_titles[idx],
                 ha="center", va="top", fontsize=10, color=node_colors,
                 linespacing=1.4)

    # 底部公式框，放在足够低的位置
    ax2.text(0, -2.1,
             "ord(σ) = lcm(3, 2, 1) = 6",
             ha="center", va="center", fontsize=12, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FDEBD0",
                       edgecolor="#E59866", linewidth=1.5))

    plt.tight_layout(pad=1.5)
    path = OUT + "cycle_demo.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {path}")



# 3. 三种映射轨迹图
def draw_map_trajectories():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x0 = 0.3141
    steps = 200

    # Logistic
    x = x0
    seq = []
    for _ in range(steps):
        x = 3.99 * x * (1 - x)
        seq.append(x)
    axes[0].plot(seq, color="#E74C3C", linewidth=0.8, alpha=0.9)
    axes[0].set_title("Logistic 映射轨迹\n($\\mu=3.99$)", fontsize=11)
    axes[0].set_xlabel("迭代步数", fontsize=10)
    axes[0].set_ylabel("$x_n$", fontsize=10)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)

    # Tent
    x = x0
    seq = []
    for _ in range(steps):
        x = 0.9999 * x if x < 0.5 else 0.9999 * (1 - x)
        seq.append(x)
    axes[1].plot(seq, color="#3498DB", linewidth=0.8, alpha=0.9)
    axes[1].set_title("Tent 映射轨迹\n($\\mu=0.9999$)", fontsize=11)
    axes[1].set_xlabel("迭代步数", fontsize=10)
    axes[1].set_ylabel("$x_n$", fontsize=10)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # Chebyshev
    x = x0
    seq = []
    for _ in range(steps):
        x = np.cos(5 * np.arccos(np.clip(x, -1, 1)))
        seq.append(x)
    axes[2].plot(seq, color="#2ECC71", linewidth=0.8, alpha=0.9)
    axes[2].set_title("Chebyshev 映射轨迹\n($k=5$)", fontsize=11)
    axes[2].set_xlabel("迭代步数", fontsize=10)
    axes[2].set_ylabel("$x_n$", fontsize=10)
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("三种混沌映射迭代轨迹（初值 $x_0=0.3141$，前200步）",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    path = OUT + "map_trajectories.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {path}")



# 4. 安全性对比雷达图
def draw_radar():
    categories = ["平均阶大小", "循环多样性", "初值敏感性", "统计均匀性", "参数健壮性"]
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    scores = {
        "Logistic":   [4.5, 4.0, 4.5, 4.0, 4.5],
        "Tent":       [1.0, 1.0, 3.5, 3.0, 2.0],
        "Chebyshev":  [3.0, 2.5, 4.0, 4.5, 3.0],
    }
    colors = {"Logistic": "#E74C3C", "Tent": "#3498DB", "Chebyshev": "#2ECC71"}

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for name, vals in scores.items():
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, "o-", linewidth=2,
                color=colors[name], label=name)
        ax.fill(angles, vals_closed, alpha=0.15, color=colors[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8, color="gray")
    ax.set_title("三种混沌映射密码学安全性综合评估", fontsize=12,
                 pad=20, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=11)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    path = OUT + "radar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {path}")



# 5. 置换矩阵热力图（对比三种映射）
def draw_permutation_matrix():
    import sys
    sys.path.insert(0, OUT)
    from chaos_permutation import (logistic_map, tent_map, chebyshev_map,
                                   generate_permutation)

    N = 64
    x0 = 0.3141

    maps = {
        "Logistic ($\\mu=3.99$)":    logistic_map(3.99, x0, 1000 + N),
        "Tent ($\\mu=0.9999$)":      tent_map(0.9999, x0, 1000 + N),
        "Chebyshev ($k=5$)":         chebyshev_map(5, x0, 1000 + N),
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (title, seq) in zip(axes, maps.items()):
        perm = generate_permutation(seq, N)
        mat = np.zeros((N, N))
        for j, i in enumerate(perm):
            mat[i, j] = 1
        ax.imshow(mat, cmap="Blues", interpolation="nearest", aspect="auto")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("输入位置", fontsize=9)
        ax.set_ylabel("输出位置", fontsize=9)
        ax.tick_params(labelsize=8)

    plt.suptitle("三种映射生成置换矩阵对比（$N=64$，黑点表示映射关系）",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    path = OUT + "perm_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {path}")


if __name__ == "__main__":
    draw_flowchart()
    draw_cycle_demo()
    draw_map_trajectories()
    draw_radar()
    draw_permutation_matrix()
