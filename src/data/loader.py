"""Owner: Person 1 (Part A). Shared loader: everyone loads the processed data through this."""
import json
from pathlib import Path

import numpy as np
import torch


def load_processed(path: str = "data/processed", device: str = "cpu") -> dict:
    """Return all processed arrays as torch tensors plus the meta dict.

    keys: train, val, test            LongTensor [M, 3]  (head, relation, tail)
          node_types                  FloatTensor [N, 8]
          edge_index, edge_type       LongTensor [2, E], [E]   train-only graph WITH inverse edges (ids 0..11)
          val_neg, test_neg           LongTensor [M, 3]  fixed negatives for AUC
          meta                        dict (num_entities, num_relations, ...)
    """
    d = Path(path)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))

    def t(name: str) -> torch.Tensor:
        return torch.from_numpy(np.load(d / f"{name}.npy")).to(device)

    return dict(train=t("train"), val=t("val"), test=t("test"), node_types=t("node_types"),
                edge_index=t("graph_edge_index"), edge_type=t("graph_edge_type"),
                val_neg=t("val_neg"), test_neg=t("test_neg"), meta=meta)
