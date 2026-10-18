"""Owner: Person 2. Unit tests for R-GCN encoder and DistMult scorer."""
from pathlib import Path
import pytest
import numpy as np
import torch

from src.encoder.rgcn import RGCNEncoder
from src.encoder.node_features import NodeFeatures
from src.encoder.train_encoder import DistMultScorer, evaluate_filtered_tail, build_known_tails_dict


@pytest.fixture
def synthetic_graph():
    """Create a deterministic synthetic graph for unit tests."""
    torch.manual_seed(42)
    num_nodes = 30
    in_dim = 72
    out_dim = 128
    num_relations = 12
    num_bases = 4

    x = torch.randn(num_nodes, in_dim)
    # Directed edges including all relations
    src = torch.randint(0, num_nodes, (80,))
    dst = torch.randint(0, num_nodes, (80,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_type = torch.randint(0, num_relations, (80,))

    return {
        "num_nodes": num_nodes,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "num_relations": num_relations,
        "num_bases": num_bases,
        "x": x,
        "edge_index": edge_index,
        "edge_type": edge_type,
    }


def test_rgcn_output_shape_synthetic(synthetic_graph):
    """Requirement 1 & 2: Output shape must be [N, out_dim]."""
    g = synthetic_graph
    encoder = RGCNEncoder(
        in_dim=g["in_dim"],
        hidden_dim=128,
        out_dim=g["out_dim"],
        num_relations=g["num_relations"],
        num_bases=g["num_bases"],
        dropout=0.2,
    )

    H = encoder(g["x"], g["edge_index"], g["edge_type"])
    assert H.shape == (g["num_nodes"], g["out_dim"]), f"Expected shape {(g['num_nodes'], g['out_dim'])}, got {H.shape}"
    assert not torch.isnan(H).any(), "Encoder output contains NaN values"


def test_rgcn_output_shape_with_dataset():
    """Requirement 2: Full graph forward pass on CropDP-KG dataset produces [11417, 128]."""
    data_dir = Path("data/processed")
    if not (data_dir / "meta.json").exists():
        pytest.skip("data/processed not found")

    from src.data.loader import load_processed
    data = load_processed("data/processed")

    node_types = data["node_types"]
    edge_index = data["edge_index"]
    edge_type = data["edge_type"]
    num_entities = data["meta"]["num_entities"]

    node_features = NodeFeatures(node_types, emb_dim=64)
    encoder = RGCNEncoder(
        in_dim=node_features.out_dim,
        hidden_dim=128,
        out_dim=128,
        num_relations=12,
        num_bases=4,
        dropout=0.2,
    )

    x = node_features()
    assert x.shape == (num_entities, 72)

    H = encoder(x, edge_index, edge_type)
    assert H.shape == (num_entities, 128), f"Expected {(num_entities, 128)}, got {H.shape}"


def test_rgcn_gradients_both_layers(synthetic_graph):
    """Requirement 7: Verify gradients flow backwards into both conv layers and embeddings."""
    g = synthetic_graph
    dummy_node_types = torch.randint(0, 2, (g["num_nodes"], 8)).float()
    node_features = NodeFeatures(dummy_node_types, emb_dim=64)

    encoder = RGCNEncoder(
        in_dim=node_features.out_dim,
        hidden_dim=128,
        out_dim=g["out_dim"],
        num_relations=g["num_relations"],
        num_bases=g["num_bases"],
        dropout=0.2,
    )

    x = node_features()
    H = encoder(x, g["edge_index"], g["edge_type"])
    loss = H.sum()
    loss.backward()

    # Check conv1 has valid gradients
    conv1_grads = [p.grad for p in encoder.conv1.parameters() if p.requires_grad]
    assert len(conv1_grads) > 0, "conv1 has no trainable parameters"
    assert all(g is not None for g in conv1_grads), "conv1 parameters did not receive gradients"
    assert any((g.abs() > 0).any() for g in conv1_grads), "conv1 gradients are all zero"

    # Check conv2 has valid gradients
    conv2_grads = [p.grad for p in encoder.conv2.parameters() if p.requires_grad]
    assert len(conv2_grads) > 0, "conv2 has no trainable parameters"
    assert all(g is not None for g in conv2_grads), "conv2 parameters did not receive gradients"
    assert any((g.abs() > 0).any() for g in conv2_grads), "conv2 gradients are all zero"

    # Check learnable embedding received gradients and registered buffer has no gradients
    assert node_features.emb.weight.grad is not None
    assert (node_features.emb.weight.grad.abs() > 0).any()
    assert node_features.types.grad is None


def test_rgcn_edge_order_invariance(synthetic_graph):
    """Requirement 7: Permuting the edge order must not change node embeddings in eval mode."""
    g = synthetic_graph
    encoder = RGCNEncoder(
        in_dim=g["in_dim"],
        hidden_dim=128,
        out_dim=g["out_dim"],
        num_relations=g["num_relations"],
        num_bases=g["num_bases"],
        dropout=0.2,
    )
    encoder.eval()

    num_edges = g["edge_index"].shape[1]
    perm = torch.randperm(num_edges)

    edge_index_perm = g["edge_index"][:, perm]
    edge_type_perm = g["edge_type"][perm]

    with torch.no_grad():
        H1 = encoder(g["x"], g["edge_index"], g["edge_type"])
        H2 = encoder(g["x"], edge_index_perm, edge_type_perm)

    diff = (H1 - H2).abs().max().item()
    assert torch.allclose(H1, H2, atol=1e-5), f"Edge order variance detected, max abs diff: {diff}"


def test_distmult_scorer_contract(synthetic_graph):
    """Requirement 3: DistMult scorer consistency between score and score_all_tails."""
    g = synthetic_graph
    H = torch.randn(g["num_nodes"], g["out_dim"])
    scorer = DistMultScorer(num_relations=6, dim=g["out_dim"])

    B = 10
    h = torch.randint(0, g["num_nodes"], (B,))
    r = torch.randint(0, 6, (B,))
    t = torch.randint(0, g["num_nodes"], (B,))

    # [B] individual triple scores
    triple_scores = scorer.score(H, h, r, t)
    assert triple_scores.shape == (B,)

    # [B, N] all candidate tails scores
    all_scores = scorer.score_all_tails(H, h, r)
    assert all_scores.shape == (B, g["num_nodes"])

    # Score of true tail from all_scores must match individual triple score
    indexed_scores = all_scores[torch.arange(B), t]
    assert torch.allclose(triple_scores, indexed_scores, atol=1e-5), "Mismatch between score() and score_all_tails()"


def test_filtered_tail_evaluation_logic():
    """Requirement 6: Filtered tail evaluation ranks true tail properly and excludes other known tails."""
    num_nodes = 5
    dim = 8
    H = torch.eye(num_nodes, dim)  # Orthogonal node representations

    scorer = DistMultScorer(num_relations=2, dim=dim)
    # Set identity relation embedding so score(h, r, t) = dot(H[h], H[t]) = 1 if h==t else 0
    with torch.no_grad():
        scorer.rel_emb.weight.fill_(1.0)

    # Suppose true triple is (0, 0, 0). True score is 1.0. Other scores are 0.0.
    # Suppose (0, 0, 1) is another known triple with high score (we force score high to test filtering)
    val_triples = torch.tensor([[0, 0, 0]])
    known_triples = np.array([[0, 0, 0], [0, 0, 1]])
    known_tails = build_known_tails_dict(known_triples)

    metrics = evaluate_filtered_tail(
        H=H,
        scorer=scorer,
        val_triples=val_triples,
        known_tails=known_tails,
        num_eval=1,
    )

    assert metrics["mrr"] == 1.0
    assert metrics["hits@1"] == 1.0
    assert metrics["hits@10"] == 1.0


def test_evaluate_filtered_tail_full_set():
    """Verify that num_eval=0 evaluates the entire set of validation triples."""
    num_nodes = 8
    dim = 16
    H = torch.randn(num_nodes, dim)
    scorer = DistMultScorer(num_relations=3, dim=dim)

    # 15 synthetic validation triples
    val_triples = torch.tensor([
        [0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 1, 4], [4, 2, 5],
        [5, 0, 6], [6, 1, 7], [7, 2, 0], [0, 1, 2], [1, 2, 3],
        [2, 1, 0], [3, 0, 1], [4, 1, 2], [5, 2, 3], [6, 0, 4],
    ])
    known_tails = build_known_tails_dict(val_triples.numpy())

    # Full set evaluation with num_eval=0
    metrics = evaluate_filtered_tail(
        H=H,
        scorer=scorer,
        val_triples=val_triples,
        known_tails=known_tails,
        num_eval=0,
    )

    assert metrics["num_eval"] == len(val_triples) == 15
    assert 0.0 < metrics["mrr"] <= 1.0
    assert 0.0 <= metrics["hits@1"] <= 1.0
    assert 0.0 <= metrics["hits@10"] <= 1.0


def test_checkpoint_metadata_fields():
    """Verify that saved checkpoint metadata contains all required run settings."""
    chk_path = Path("results/encoder_checkpoint_full_val.pt")
    if not chk_path.exists():
        chk_path = Path("results/encoder_checkpoint.pt")
    if not chk_path.exists():
        pytest.skip("No checkpoint found in results/")

    ckpt = torch.load(chk_path, weights_only=False)
    required_keys = [
        "epoch", "actual_epochs_completed", "lr", "num_neg",
        "seed", "eval_limit", "metrics", "config",
        "node_features_state_dict", "encoder_state_dict", "scorer_state_dict",
    ]
    for key in required_keys:
        assert key in ckpt, f"Missing required metadata key: {key}"

    assert isinstance(ckpt["metrics"], dict)
    assert "mrr" in ckpt["metrics"]
    assert "hits@10" in ckpt["metrics"]
    assert ckpt["metrics"]["num_eval"] > 0


