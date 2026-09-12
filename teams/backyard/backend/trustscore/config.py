"""All tunable / hard-coded values of the pipeline live here so the design doc can list them in one table."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class DataConfig:
    # External (unlabelled) accounts that touch a single dataset account carry no cross-account information
    # (they can only echo their unique neighbour), so they are pruned. 2 keeps every shared neighbour.
    min_external_degree: int = 2


@dataclass
class PropagationConfig:
    # r = (1-alpha) q + alpha P r ; alpha = share of a node's residual that comes from its neighbourhood.
    alpha: float = 0.8
    # Prior residual of a labelled seed: q = +-seed_residual (0.5 = certain; smaller tolerates label noise).
    seed_residual: float = 0.5
    # Confidence given to the local (profile) model for unlabelled nodes: q = lam * (p_local - 0.5).
    lam: float = 0.5
    # A sender u is damped by 1/deg(u)^hub_damping: endorsements from mass-followers / celebrities count less.
    hub_damping: float = 0.5
    # Relative homophily strength of each edge channel (gather side, "v collects from u").
    relation_weights: dict[str, float] = field(default_factory=lambda: {
        "mutual": 1.0,        # u and v follow each other: strongest social tie
        "follow_in": 0.5,     # u follows v (one-way): endorsement / amplification of v by u
        "follow_out": 0.25,   # v follows u (one-way): weak (anyone can follow anyone)
        "mention_in": 0.5, "mention_out": 0.5,
        "reply_in": 0.5, "reply_out": 0.5,
        "quoted_in": 0.5, "quoted_out": 0.5,
        "url": 0.25, "hashtag": 0.25,   # co-behaviour relations (MGTAB): weak, symmetric
    })
    max_iter: int = 300
    tol: float = 1e-8


@dataclass
class LocalModelConfig:
    n_estimators: int = 500
    min_samples_leaf: int = 2
    max_features: str = "sqrt"
    class_weight: str = "balanced_subsample"
    random_state: int = 0
    use_content_features: bool = False  # MGTAB only: 768-d tweet embeddings are excluded by design (judge the source, not the text)


@dataclass
class ClusterConfig:
    similarity_top_k: int = 10        # co-following similarity graph: keep the k most similar accounts per account
    similarity_min: float = 0.05      # ... and only if cosine similarity >= this
    leiden_resolution: float = 1.0
    leiden_seed: int = 0
    fraudar_blocks: int = 3           # number of dense blocks peeled off greedily
    burst_window_days: int = 1        # feature 'created_within_window': dataset accounts created within +-window days
    burst_min_accounts: int = 20      # burst flag: a calendar day on which at least this many dataset accounts were created
    hub_display_count: int = 150      # external hubs shown in the UI graph
    display_similarity_top_k: int = 5


@dataclass
class EvalConfig:
    seed_fractions: tuple = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
    repeats: int = 5
    noise_levels: tuple = (0.0, 0.1, 0.2, 0.3)
    noise_seed_fraction: float = 0.10
    crossfit_folds: int = 5
    random_state: int = 0


@dataclass
class Config:
    dataset: str = "cresci-2015"
    data: DataConfig = field(default_factory=DataConfig)
    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    local_model: LocalModelConfig = field(default_factory=LocalModelConfig)
    clusters: ClusterConfig = field(default_factory=ClusterConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        return asdict(self)
