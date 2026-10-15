"""Owner: Person 4 (Part D). Baselines.
Each must expose make_score_all_tails() -> score_all_tails(h, r) -> [B, N].  TODO."""


class PopularityBaseline:
    """Sanity check: always rank tails by how often they appear as the tail of relation r in train."""


class RotatEBaseline:
    """RotatE alone with plain entity embeddings (torch_geometric.nn.kge.RotatE or own code)."""


class RGCNDistMultBaseline:
    """R-GCN encoder + DistMult decoder (no variational part)."""
