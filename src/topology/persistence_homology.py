from gtda.time_series import SingleTakensEmbedding
from gtda.plotting import plot_point_cloud
from gtda.homology import VietorisRipsPersistence
import numpy as np
from typing import Callable
from sklearn.decomposition import PCA
from gtda.diagrams import PersistenceEntropy

def _get_embedder_time_series(max_embedding_dim: int = 10, max_time_delay: int = 10 , stride: int=5) -> SingleTakensEmbedding:
    embedder = SingleTakensEmbedding(
        parameters_type = "search",
        n_jobs = -1,
        time_delay = max_time_delay,
        dimension = max_embedding_dim,
        stride = stride
    )
    return embedder

def _fit_embedder(embedder: SingleTakensEmbedding, y: np.ndarray, verbose: bool=False) -> np.ndarray:
    """Fits a Takens embedder and displays optimal search parameters."""
    y_embedded = embedder.fit_transform(y)

    if verbose:
        print(f"Shape of embedded time series: {y_embedded.shape}")
        print(
            f"Optimal embedding dimension is {embedder.dimension_} and time delay is {embedder.time_delay_}"
        )

    return y_embedded


def embed_time_series(data: np.ndarray, embedder: Callable =    _get_embedder_time_series, fit_func: Callable = _fit_embedder):
    embedder = embedder()
    ts_embedded = fit_func(embedder, data)

    return ts_embedded


def plot_pcd(data: list[float], plot_name: str) -> None:
    """Plot and save pcd"""
    plot_point_cloud(data).show()


def reduce_dimension(data: np.ndarray, n_components: int =3) -> np.ndarray:
    pca = PCA(n_components=n_components)
    ts_embedded_pca = pca.fit_transform(data)
    return ts_embedded_pca



def vietoris_rips_transform(data: list[float], symbol: str) -> np.ndarray:
    data = data[None, :, :] # reshape for analysis
    # 0 - connected components, 1 - loops, 2 - voids
    homology_dimensions = [0, 1, 2]

    persistence = VietorisRipsPersistence(
        homology_dimensions=homology_dimensions, n_jobs=6
    )

    persistence_diagram = persistence.fit_transform(data)

    return persistence_diagram


def persistence_entropy(persistence_diagram: np.ndarray) -> np.ndarray:
    persistence_entropy = PersistenceEntropy()

    # calculate topological feature matrix
    pe = persistence_entropy.fit_transform(persistence_diagram)

    # expect shape - (n_point_clouds, n_homology_dims)
    return pe


