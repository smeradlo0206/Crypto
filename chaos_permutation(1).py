"""
混沌置乱的循环阶分析
三种混沌映射：Logistic、Tent、Chebyshev
两种置乱方法：argsort 排序法、混沌 Fisher-Yates 法
分析置乱表的循环结构，绘制"平均阶-N"曲线
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_font_path = "/System/Library/Fonts/Supplemental/Songti.ttc"
fm.fontManager.addfont(_font_path)
_prop = fm.FontProperties(fname=_font_path)
plt.rcParams["font.family"] = _prop.get_name()
plt.rcParams["axes.unicode_minus"] = False
from math import gcd
from functools import reduce
from collections import Counter


# ──────────────────────────────────────────────
# 1. 混沌映射定义
# ──────────────────────────────────────────────

def logistic_map(mu, x0, steps):
    """Logistic 映射：x_{n+1} = mu * x_n * (1 - x_n)，mu ∈ (3.57, 4)"""
    x = x0
    seq = []
    for _ in range(steps):
        x = mu * x * (1 - x)
        seq.append(x)
    return seq


def tent_map(mu, x0, steps):
    """Tent 映射：x_{n+1} = mu*x if x<0.5 else mu*(1-x)"""
    x = x0
    seq = []
    for _ in range(steps):
        x = mu * x if x < 0.5 else mu * (1 - x)
        seq.append(x)
    return seq


def chebyshev_map(k, x0, steps):
    """Chebyshev 映射：x_{n+1} = cos(k * arccos(x_n))，x ∈ [-1, 1]，k≥2"""
    x = x0
    seq = []
    for _ in range(steps):
        x = np.cos(k * np.arccos(np.clip(x, -1, 1)))
        seq.append(x)
    return seq


# ──────────────────────────────────────────────
# 2. 置乱表生成：两种方法
# ──────────────────────────────────────────────

def generate_permutation_argsort(chaotic_seq, N):
    """
    方法一：argsort 排序法
    取混沌序列后 N 项，按升序排列，原始下标即为置换。
    """
    seq = np.array(chaotic_seq[-N:])
    return np.argsort(seq)


def generate_permutation_fy(chaotic_seq, N):
    """
    方法二：混沌 Fisher-Yates 洗牌法
    用混沌序列驱动 Knuth shuffle：
    对 i = N-1, N-2, ..., 1，将位置 i 与位置
    j = floor(x_i * (i+1)) 交换，其中 x_i 来自混沌序列。
    消耗 N 个混沌值（取序列后 N 项）。
    """
    seq = np.array(chaotic_seq[-N:])
    perm = np.arange(N)
    for i in range(N - 1, 0, -1):
        j = int(seq[N - 1 - i] * (i + 1)) % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def generate_permutation(chaotic_seq, N, method="argsort"):
    """统一接口：method='argsort' 或 'fy'"""
    if method == "argsort":
        return generate_permutation_argsort(chaotic_seq, N)
    else:
        return generate_permutation_fy(chaotic_seq, N)


# ──────────────────────────────────────────────
# 3. 循环结构分析
# ──────────────────────────────────────────────

def analyze_cycles(perm):
    """
    分析置换的循环结构。
    返回 cycle_lengths, order, Counter({长度: 数量})
    """
    N = len(perm)
    visited = np.zeros(N, dtype=bool)
    cycle_lengths = []
    for start in range(N):
        if visited[start]:
            continue
        length = 0
        cur = start
        while not visited[cur]:
            visited[cur] = True
            cur = perm[cur]
            length += 1
        cycle_lengths.append(length)
    order = reduce(lambda a, b: a * b // gcd(a, b), cycle_lengths)
    return cycle_lengths, order, Counter(cycle_lengths)


# ──────────────────────────────────────────────
# 4. 多种子平均阶（支持参数和方法选择）
# ──────────────────────────────────────────────

def _make_seq(map_name, param, x0, N):
    """生成混沌序列（预热1000步）"""
    steps = 1000 + N
    if map_name == "Logistic":
        return logistic_map(param, x0, steps)
    elif map_name == "Tent":
        return tent_map(param, x0, steps)
    elif map_name == "Chebyshev":
        return chebyshev_map(int(param), x0, steps)
    raise ValueError(f"未知映射: {map_name}")


def mean_order(map_name, param, N, n_seeds=30, seed_base=42, method="argsort"):
    """估计给定映射、参数、置换长度下的平均阶"""
    rng = np.random.default_rng(seed_base)
    orders = []
    for _ in range(n_seeds):
        x0 = rng.uniform(-0.999, 0.999) if map_name == "Chebyshev" \
             else rng.uniform(0.001, 0.999)
        seq = _make_seq(map_name, param, x0, N)
        perm = generate_permutation(seq, N, method=method)
        _, order, _ = analyze_cycles(perm)
        orders.append(order)
    return np.mean(orders)


# ──────────────────────────────────────────────
# 5. 理论参考线
# ──────────────────────────────────────────────

def ref_order(N):
    """
    随机置换平均阶的精确渐近式（Erdos-Turan 定理）：
        E[ord(sigma_N)] ~ exp(sqrt(N * ln N))
    注：e*ln(N) 是对该式在小 N 下的粗略简化，不反映真实渐近行为，
    此处使用正确渐近式作为参考。
    """
    return np.exp(np.sqrt(N * np.log(N)))


# ──────────────────────────────────────────────
# 6. 单次演示
# ──────────────────────────────────────────────

def demo_single(map_name, param, N=64, x0=0.3141, method="argsort"):
    seq = _make_seq(map_name, param, x0, N)
    perm = generate_permutation(seq, N, method=method)
    cycle_lengths, order, counter = analyze_cycles(perm)
    print(f"\n{'='*55}")
    print(f"映射: {map_name}(param={param})  N={N}  方法={method}  x0={x0}")
    print(f"循环长度分布: {dict(sorted(counter.items()))}")
    print(f"循环总数: {len(cycle_lengths)}")
    print(f"置换的阶: {order}")
    return perm, cycle_lengths, order


# ──────────────────────────────────────────────
# 7. 绘图函数
# ──────────────────────────────────────────────

def plot_cycle_distribution(map_name, param, N=128, x0=0.3141,
                            method="argsort", save_path=None):
    """循环长度分布柱状图"""
    seq = _make_seq(map_name, param, x0, N)
    perm = generate_permutation(seq, N, method=method)
    _, _, counter = analyze_cycles(perm)
    lengths = sorted(counter.keys())
    counts = [counter[l] for l in lengths]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(l) for l in lengths], counts, color="steelblue", edgecolor="white")
    ax.set_xlabel("循环长度", fontsize=12)
    ax.set_ylabel("循环数量", fontsize=12)
    ax.set_title(f"{map_name}(param={param}) 循环长度分布（N={N}，{method}法）", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"已保存: {save_path}")
    plt.close()


def plot_avg_order_vs_N(N_list, n_seeds=30, save_path=None):
    """
    主曲线：三种映射（各一组代表性参数）平均阶 vs N
    参考线：exp(sqrt(N * ln N))  —— 随机置换平均阶渐近式
    注：早期文献有时用 e*ln(N) 作粗略参考，但该式仅在极小 N
    下接近真实值；本图采用正确渐近式 exp(sqrt(N*ln N))。
    """
    configs = [
        ("Logistic",   3.99,   "#e74c3c", "Logistic (mu=3.99)"),
        ("Tent",       0.9999, "#2ecc71", "Tent (mu=0.9999)"),
        ("Chebyshev",  5,      "#3498db", "Chebyshev (k=5)"),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    for map_name, param, color, label in configs:
        print(f"计算 {label} 平均阶...")
        avg = [mean_order(map_name, param, N, n_seeds=n_seeds) for N in N_list]
        ax.plot(N_list, avg, "o-", label=label, color=color,
                linewidth=2, markersize=5)

    ref = [ref_order(N) for N in N_list]
    ax.plot(N_list, ref, "k--", linewidth=1.8, alpha=0.7,
            label=r"理论参考：$\exp(\sqrt{N\ln N})$")

    ax.set_xlabel("置换长度 N", fontsize=12)
    ax.set_ylabel("平均阶（对数轴）", fontsize=12)
    ax.set_yscale("log")
    ax.set_title("三种混沌映射置乱表平均阶 vs N（argsort 法）", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"已保存: {save_path}")
    plt.close()


def plot_multi_param(map_name, param_list, N_list, n_seeds=20,
                     method="argsort", save_path=None):
    """
    固定映射类型，对比多组参数下平均阶-N 曲线。
    同时画出随机置换理论参考线。
    """
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, param in enumerate(param_list):
        print(f"  计算 {map_name}(param={param})...")
        avg = [mean_order(map_name, param, N, n_seeds=n_seeds, method=method)
               for N in N_list]
        ax.plot(N_list, avg, "o-", color=cmap(i),
                label=f"param={param}", linewidth=2, markersize=5)

    ref = [ref_order(N) for N in N_list]
    ax.plot(N_list, ref, "k--", linewidth=1.8, alpha=0.7,
            label=r"理论参考：$\exp(\sqrt{N\ln N})$")

    ax.set_xlabel("置换长度 N", fontsize=12)
    ax.set_ylabel("平均阶（对数轴）", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(f"{map_name} 映射不同参数下平均阶 vs N", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"已保存: {save_path}")
    plt.close()


def plot_method_comparison(map_name, param, N_list, n_seeds=30, save_path=None):
    """
    对比 argsort 法与 Fisher-Yates 法在相同映射、相同参数下的平均阶。
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, color, ls in [("argsort", "#e74c3c", "-"),
                               ("fy",      "#3498db", "--")]:
        label = "argsort 排序法" if method == "argsort" else "Fisher-Yates 洗牌法"
        print(f"  计算 {method} 法...")
        avg = [mean_order(map_name, param, N, n_seeds=n_seeds, method=method)
               for N in N_list]
        ax.plot(N_list, avg, f"o{ls}", color=color, label=label,
                linewidth=2, markersize=5)

    ref = [ref_order(N) for N in N_list]
    ax.plot(N_list, ref, "k--", linewidth=1.8, alpha=0.7,
            label=r"理论参考：$\exp(\sqrt{N\ln N})$")

    ax.set_xlabel("置换长度 N", fontsize=12)
    ax.set_ylabel("平均阶（对数轴）", fontsize=12)
    ax.set_yscale("log")
    ax.set_title(f"{map_name}(param={param}) 两种置乱方法平均阶对比", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"已保存: {save_path}")
    plt.close()


def plot_order_histogram(map_name, param, N=64, n_seeds=200,
                         method="argsort", save_path=None):
    """固定 N 下阶的分布直方图"""
    rng = np.random.default_rng(0)
    orders = []
    for _ in range(n_seeds):
        x0 = rng.uniform(-0.999, 0.999) if map_name == "Chebyshev" \
             else rng.uniform(0.001, 0.999)
        seq = _make_seq(map_name, param, x0, N)
        perm = generate_permutation(seq, N, method=method)
        _, order, _ = analyze_cycles(perm)
        orders.append(order)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(orders, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(np.mean(orders), color="red", linestyle="--",
               label=f"均值={np.mean(orders):.1f}")
    ax.set_xlabel("阶", fontsize=12)
    ax.set_ylabel("频次", fontsize=12)
    ax.set_title(f"{map_name}(param={param}) 置换阶分布"
                 f"（N={N}，{n_seeds}种子，{method}法）", fontsize=11)
    ax.legend(fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"已保存: {save_path}")
    plt.close()


# ──────────────────────────────────────────────
# 8. 主程序
# ──────────────────────────────────────────────

if __name__ == "__main__":
    OUT = "/Users/kai/Desktop/practice/2026/05/30__SquidwardCoIfan/"

    # ── 单次演示（代表性参数）──
    demo_single("Logistic",   3.99,   N=64)
    demo_single("Tent",       0.9999, N=64)
    demo_single("Chebyshev",  5,      N=64)

    # ── 循环分布柱状图 ──
    for name, param in [("Logistic", 3.99), ("Tent", 0.9999), ("Chebyshev", 5)]:
        plot_cycle_distribution(name, param, N=128,
                                save_path=OUT + f"cycle_dist_{name}.png")

    # ── 阶分布直方图（N=64）──
    for name, param in [("Logistic", 3.99), ("Tent", 0.9999), ("Chebyshev", 5)]:
        plot_order_histogram(name, param, N=64, n_seeds=200,
                             save_path=OUT + f"order_hist_{name}.png")

    # ── 主平均阶-N 曲线（N 扩展到 512，覆盖常见图像行数）──
    N_list_main = [10, 20, 50, 100, 128, 200, 256, 320, 400, 512]
    plot_avg_order_vs_N(N_list_main, n_seeds=30,
                        save_path=OUT + "avg_order_vs_N.png")

    # ── Logistic 多参数对比 ──
    print("\nLogistic 多参数对比...")
    N_list_short = [10, 20, 50, 100, 128, 200, 256]
    plot_multi_param("Logistic",
                     param_list=[3.70, 3.90, 3.99],
                     N_list=N_list_short, n_seeds=20,
                     save_path=OUT + "multi_param_Logistic.png")

    # ── Chebyshev 多参数对比 ──
    print("\nChebyshev 多参数对比...")
    plot_multi_param("Chebyshev",
                     param_list=[3, 5, 7, 10],
                     N_list=N_list_short, n_seeds=20,
                     save_path=OUT + "multi_param_Chebyshev.png")

    # ── 两种置乱方法对比（Logistic mu=3.99）──
    print("\n两种置乱方法对比...")
    plot_method_comparison("Logistic", 3.99,
                           N_list=N_list_short, n_seeds=30,
                           save_path=OUT + "method_comparison.png")

    print("\n全部完成！")
