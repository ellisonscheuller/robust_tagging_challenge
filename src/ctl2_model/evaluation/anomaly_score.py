import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance


class ClassifierAnomalyScore:
    def __init__(self, encoder, normal_class_idx: int = 2):
        self.encoder = encoder
        self.normal_class_idx = normal_class_idx

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = self.encoder.predict(x, batch_size=2048, verbose=0)
        probs = np.asarray(out["class_probs"])
        return 1.0 - probs[:, self.normal_class_idx]


class KNNAnomalyScore:
    def __init__(self, encoder, k: int = 5):
        self.encoder = encoder
        self.k = k
        self.knn = None

    def fit(self, X_normal: np.ndarray):
        latents = self.encoder.predict(X_normal, batch_size=2048, verbose=0)["latent"]
        latents_norm = normalize(latents, axis=1)
        self.knn = NearestNeighbors(n_neighbors=self.k, metric="euclidean", n_jobs=-1)
        self.knn.fit(latents_norm)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        latents = self.encoder.predict(x, batch_size=2048, verbose=0)["latent"]
        latents_norm = normalize(latents, axis=1)
        dists, _ = self.knn.kneighbors(latents_norm)
        return dists[:, -1]


class MahalanobisAnomalyScore:
    def __init__(self, encoder, pca_dim: int = 16):
        self.encoder = encoder
        self.pca_dim = pca_dim
        self.pca = None
        self.mu = None
        self.Sigma_inv = None
        self.logdet = None
        self.D = None

    def fit(self, X_normal: np.ndarray):
        latents = self.encoder.predict(X_normal, batch_size=2048, verbose=0)["latent"]
        pca_dim = min(self.pca_dim, latents.shape[1])
        self.pca = PCA(n_components=pca_dim, whiten=False)
        Z = self.pca.fit_transform(latents)
        cov = EmpiricalCovariance().fit(Z)
        self.mu = cov.location_
        self.Sigma_inv = np.linalg.inv(cov.covariance_)
        sign, self.logdet = np.linalg.slogdet(cov.covariance_)
        self.D = Z.shape[1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        latents = self.encoder.predict(x, batch_size=2048, verbose=0)["latent"]
        Z = self.pca.transform(latents)
        diff = Z - self.mu
        d2 = np.einsum("bi,ij,bj->b", diff, self.Sigma_inv, diff)
        log_prob = -0.5 * (d2 + self.logdet + self.D * np.log(2 * np.pi))
        return -log_prob


class CombinedAnomalyScore:
    def __init__(
        self,
        encoder,
        normal_class_idx: int = 2,
        knn_k: int = 5,
        knn_lambda: float = 0.5,
    ):
        self.classifier = ClassifierAnomalyScore(encoder, normal_class_idx)
        self.knn = KNNAnomalyScore(encoder, knn_k)
        self.knn_lambda = knn_lambda
        self.knn_min = None
        self.knn_max = None

    def fit(self, X_normal: np.ndarray):
        self.knn.fit(X_normal)
        knn_normal = self.knn(X_normal)
        self.knn_min = knn_normal.min()
        self.knn_max = knn_normal.max()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        c_score = self.classifier(x)
        k_score = self.knn(x)
        k_norm = (k_score - self.knn_min) / (self.knn_max - self.knn_min + 1e-8)
        return c_score + self.knn_lambda * k_norm


BUNCH_CROSSING_RATE = 40e3
FILLED_BUNCHES = 2760 / 3564


def get_rate_curve(scores_sig, scores_bg, thresholds):
    bg_rate = np.array([(scores_bg > t).mean() for t in thresholds])
    sig_eff = np.array([(scores_sig > t).mean() for t in thresholds])
    bg_rate_khz = bg_rate * BUNCH_CROSSING_RATE * FILLED_BUNCHES
    return bg_rate_khz, sig_eff


def efficiency_at_rate(scores_sig, scores_bg, target_rate_hz: float = 10e3):
    effective_rate = BUNCH_CROSSING_RATE * FILLED_BUNCHES
    target_fpr = target_rate_hz / effective_rate
    thresholds = np.percentile(scores_bg, np.linspace(0, 100, 10000))
    bkg_rates = np.array([(scores_bg > t).mean() for t in thresholds])
    idx = np.argmin(np.abs(bkg_rates - target_fpr))
    return float((scores_sig > thresholds[idx]).mean())
