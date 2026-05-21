"""
构型拓扑表征：基于 Delaunay 三角剖分的离子晶格构型识别

将离子位置转化为与标签无关的拓扑表示（邻接图 + 配位数指纹），
用于构型比较和结构重构检测。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, Delaunay


@dataclass(frozen=True)
class TopologyFingerprint:
    """构型拓扑指纹，用于快速比较"""
    coord_seq: tuple[int, ...]      # 按升序排列的配位数序列
    n_boundary: int                 # 边界（凸包）离子数


@dataclass
class CrystalTopology:
    """完整的晶格拓扑"""
    adj_matrix: np.ndarray          # (N, N) bool 邻接矩阵
    adjacency: list[list[int]]      # 各离子的邻居索引列表
    coord_numbers: np.ndarray       # (N,) 各离子配位数
    boundary: np.ndarray            # 边界离子索引
    fingerprint: TopologyFingerprint
    plane: str = ""                 # 投影平面标识


def project_to_plane(r_um: np.ndarray, plane: str = "xoz") -> np.ndarray:
    """将 3D 位置投影到 2D 平面

    Parameters
    ----------
    r_um : (N, 3)
    plane : 'xoz' | 'yoz' | 'xoy' | 'auto'
        'auto' 使用 PCA 选择方差最大的两个轴
    """
    r_um = np.asarray(r_um, dtype=float)
    if plane == "xoz":
        return r_um[:, [0, 2]]
    if plane == "yoz":
        return r_um[:, [1, 2]]
    if plane == "xoy":
        return r_um[:, [0, 1]]
    if plane == "auto":
        centered = r_um - r_um.mean(axis=0)
        cov = centered.T @ centered
        _, eigvecs = np.linalg.eigh(cov)
        return centered @ eigvecs[:, -2:]
    raise ValueError(f"未知投影平面: {plane}")


def _extract_edges(simplices: np.ndarray) -> set[tuple[int, int]]:
    """从三角剖分的单纯形中提取去重边集"""
    edges: set[tuple[int, int]] = set()
    for tri in simplices:
        for i in range(len(tri)):
            for j in range(i + 1, len(tri)):
                edge = (int(min(tri[i], tri[j])), int(max(tri[i], tri[j])))
                edges.add(edge)
    return edges


def _filter_edges_by_distance(
    edges: set[tuple[int, int]],
    points_2d: np.ndarray,
    factor: float = 1.5,
) -> set[tuple[int, int]]:
    """过滤过长边：仅保留长度 < factor × 平均最近邻距离的边

    用于去除 Delaunay 在准一维链（线性链）上产生的伪边。
    """
    dist_mat = np.linalg.norm(points_2d[:, None, :] - points_2d[None, :, :], axis=2)
    np.fill_diagonal(dist_mat, np.inf)
    nn_dist = np.min(dist_mat, axis=1).mean()
    threshold = nn_dist * factor

    return {
        (i, j) for i, j in edges
        if dist_mat[i, j] < threshold
    }


def _build_topology_1d(
    r_um: np.ndarray,
    points_2d: np.ndarray,
    plane: str,
) -> CrystalTopology:
    """共线退化处理：沿主方向排序，相邻离子互为邻居"""
    n = r_um.shape[0]
    centered = points_2d - points_2d.mean(axis=0)
    direction = centered[0] if n == 1 else np.linalg.svd(centered, compute_uv=False)
    if n <= 1:
        pass
    # 沿主轴投影排序
    if n > 1:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ vt[0]
    else:
        proj = np.array([0.0])
    order = np.argsort(proj)

    edges: set[tuple[int, int]] = set()
    for k in range(len(order) - 1):
        i, j = int(order[k]), int(order[k + 1])
        edges.add((min(i, j), max(i, j)))

    # 边界 = 链的两端
    boundary = np.array([int(order[0]), int(order[-1])], dtype=int)

    adj_matrix = np.zeros((n, n), dtype=bool)
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for i, j in edges:
        adj_matrix[i, j] = True
        adj_matrix[j, i] = True
        adjacency[i].append(j)
        adjacency[j].append(i)

    coord_numbers = np.array([len(nb) for nb in adjacency], dtype=int)
    fingerprint = TopologyFingerprint(
        coord_seq=tuple(sorted(coord_numbers.tolist())),
        n_boundary=int(len(boundary)),
    )
    return CrystalTopology(
        adj_matrix=adj_matrix, adjacency=adjacency,
        coord_numbers=coord_numbers, boundary=boundary,
        fingerprint=fingerprint, plane=plane,
    )


def build_topology(
    r_um: np.ndarray,
    plane: str = "xoz",
    edge_filter_factor: float = 1.5,
) -> CrystalTopology:
    """从离子位置构建晶格拓扑

    Parameters
    ----------
    r_um : (N, 3)
        离子位置，单位 μm
    plane : str
        投影平面，默认 'xoz'（线性阱的轴向 z + 径向 x）
    edge_filter_factor : float
        边长过滤因子，保留长度 < factor × 平均最近邻距离的边。
        设为 0 禁用过滤。
    """
    r_um = np.asarray(r_um, dtype=float)
    n = r_um.shape[0]
    points_2d = project_to_plane(r_um, plane)

    # 检测共线退化：若投影后所有点在一条直线上，退化为 1D 邻接
    centered = points_2d - points_2d.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    is_collinear = singular_values[1] < 1e-10 * singular_values[0] if n > 2 else True

    if is_collinear:
        return _build_topology_1d(r_um, points_2d, plane)

    hull = ConvexHull(points_2d)
    boundary = np.asarray(hull.vertices, dtype=int)

    tri = Delaunay(points_2d)
    edges = _extract_edges(tri.simplices)

    if edge_filter_factor > 0:
        edges = _filter_edges_by_distance(edges, points_2d, edge_filter_factor)

    adj_matrix = np.zeros((n, n), dtype=bool)
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for i, j in edges:
        adj_matrix[i, j] = True
        adj_matrix[j, i] = True
        adjacency[i].append(j)
        adjacency[j].append(i)

    coord_numbers = np.array([len(neighbors) for neighbors in adjacency], dtype=int)
    coord_seq = tuple(sorted(coord_numbers.tolist()))

    fingerprint = TopologyFingerprint(
        coord_seq=coord_seq,
        n_boundary=int(len(boundary)),
    )

    return CrystalTopology(
        adj_matrix=adj_matrix,
        adjacency=adjacency,
        coord_numbers=coord_numbers,
        boundary=boundary,
        fingerprint=fingerprint,
        plane=plane,
    )


def same_topology(t1: CrystalTopology, t2: CrystalTopology) -> bool:
    """比较两个拓扑是否等价

    快速路径：指纹不同 → 一定不等价（O(N log N)）
    慢速路径：指纹相同 → 比较邻接矩阵的图同构性
    """
    if t1.fingerprint != t2.fingerprint:
        return False
    n1, n2 = t1.adj_matrix.shape[0], t2.adj_matrix.shape[0]
    if n1 != n2:
        return False

    # 指纹相同，进一步比较邻接矩阵
    # 先检查每行的度数排序是否完全一致
    deg1 = sorted(t1.adj_matrix.sum(axis=1).tolist())
    deg2 = sorted(t2.adj_matrix.sum(axis=1).tolist())
    if deg1 != deg2:
        return False

    # 度数完全一致，比较谱（邻接矩阵特征值）作为图同构的近似判据
    eig1 = np.sort(np.linalg.eigvalsh(t1.adj_matrix.astype(float)))
    eig2 = np.sort(np.linalg.eigvalsh(t2.adj_matrix.astype(float)))
    return bool(np.allclose(eig1, eig2, atol=1e-8))


def topology_distance(t1: CrystalTopology, t2: CrystalTopology) -> float:
    """拓扑距离：基于配位数分布的 Wasserstein-1 距离"""
    c1 = np.array(t1.fingerprint.coord_seq, dtype=float)
    c2 = np.array(t2.fingerprint.coord_seq, dtype=float)
    if len(c1) != len(c2):
        return float("inf")
    return float(np.sum(np.abs(np.sort(c1) - np.sort(c2))))
