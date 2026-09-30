import numpy as np
import numpy.typing as npt

class Model:
    """An ERGM for networks with ``n`` vertices, implemented in Rust."""

    def __init__(
        self, n: int, directed: bool, terms: list[tuple[str, list[float], list[int]]]
    ) -> None: ...
    @property
    def n_stats(self) -> int: ...
    def summary(self, edges: npt.NDArray[np.uint32]) -> list[float]: ...
    def mple_data(
        self, edges: npt.NDArray[np.uint32]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def simulate(
        self,
        starts: list[npt.NDArray[np.uint32]],
        theta: list[float],
        burnin: int,
        interval: int,
        samplesize: int,
        seed: int,
        keep_networks: bool = False,
        triadic_weight: float = 0.0,
    ) -> tuple[
        npt.NDArray[np.float64],
        list[npt.NDArray[np.uint32]],
        list[list[npt.NDArray[np.uint32]]],
    ]: ...
