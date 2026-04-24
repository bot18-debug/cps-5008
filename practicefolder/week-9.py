from typing import List
import math


class DBSCANClustering:
    def __init__(self, eps: float, min_pts: int) -> None:
        self._eps: float = eps
        self._min_pts: int = min_pts
        self._labels: List[int] = []
        self._visited: List[bool] = []
        self._cluster_id: int = 0

    def _distance(self, p1: List[float], p2: List[float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def _region_query(self, data: List[List[float]], point_idx: int) -> List[int]:
        neighbours: List[int] = []
        for i, point in enumerate(data):
            if self._distance(data[point_idx], point) <= self._eps:
                neighbours.append(i)
        return neighbours

    def _expand_cluster(
        self,
        data: List[List[float]],
        point_idx: int,
        neighbours: List[int],
        cluster_id: int
    ) -> None:
        self._labels[point_idx] = cluster_id

        i = 0
        while i < len(neighbours):
            neighbour_idx = neighbours[i]

            if not self._visited[neighbour_idx]:
                self._visited[neighbour_idx] = True
                new_neighbours = self._region_query(data, neighbour_idx)

                if len(new_neighbours) >= self._min_pts:
                    neighbours.extend(new_neighbours)

            if self._labels[neighbour_idx] == -1:
                self._labels[neighbour_idx] = cluster_id

            i += 1

    def fit(self, data: List[List[float]]) -> None:
        n = len(data)
        self._labels = [-1] * n  
        self._visited = [False] * n
        self._cluster_id = 0

        for i in range(n):
            if self._visited[i]:
                continue

            self._visited[i] = True
            neighbours = self._region_query(data, i)

            if len(neighbours) < self._min_pts:
                self._labels[i] = -1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
            else:
                self._expand_cluster(data, i, neighbours, self._cluster_id)
                self._cluster_id += 1

    def get_labels(self) -> List[int]:
        return self._labels

    def get_clusters(self, data: List[List[float]]) -> List[List[List[float]]]:
        clusters = {}
        for label, point in zip(self._labels, data):
            if label == -1:
                continue
            clusters.setdefault(label, []).append(point)
        return list(clusters.values())

    def get_noise(self, data: List[List[float]]) -> List[List[float]]:
        return [point for label, point in zip(self._labels, data) if label == -1]