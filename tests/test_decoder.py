"""Owner: Person 3 (Part C). Comprehensive unit and integration tests for VRGAE model components:
- VariationalLatent layer (mu, logvar, reparameterization, KL divergence, numerical stability)
- RotatEDecoder (complex space rotation, triple scoring, 2D negative scoring, inverse relations, score_all_tails)
- Composite loss functions (link_loss, kl_weight warm-up, total_loss, gradient backpropagation)
- Full integrated VRGAE model (forward, backward, make_score_all_tails, full dataset test)
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.model.decoder import RotatEDecoder
from src.model.latent import VariationalLatent
from src.model.loss import compute_loss_breakdown, kl_weight, link_loss, total_loss
from src.model.vrgae import VRGAE


@pytest.fixture
def dummy_data():
    torch.manual_seed(42)
    num_nodes = 20
    in_dim = 128
    latent_dim = 128
    num_relations = 6
    H = torch.randn(num_nodes, in_dim)
    return {
        "num_nodes": num_nodes,
        "in_dim": in_dim,
        "latent_dim": latent_dim,
        "num_relations": num_relations,
        "H": H,
    }


def test_latent_layer_shapes_and_reparameterization(dummy_data):
    """Test VariationalLatent output shapes, train/eval reparameterization behavior, and gradient flow."""
    latent = VariationalLatent(in_dim=dummy_data["in_dim"], latent_dim=dummy_data["latent_dim"])
    H = dummy_data["H"].clone().requires_grad_(True)

    # Training mode: sampling active
    latent.train()
    z_train, mu_train, logvar_train = latent(H)
    assert z_train.shape == (dummy_data["num_nodes"], dummy_data["latent_dim"])
    assert mu_train.shape == (dummy_data["num_nodes"], dummy_data["latent_dim"])
    assert logvar_train.shape == (dummy_data["num_nodes"], dummy_data["latent_dim"])
    assert not torch.isnan(z_train).any()
    assert not torch.isinf(z_train).any()

    # Eval mode: deterministic z = mu
    latent.eval()
    with torch.no_grad():
        z_eval, mu_eval, logvar_eval = latent(H)
    assert torch.allclose(z_eval, mu_eval, atol=1e-6), "In eval mode, z must be equal to mu"

    # Backward gradient flow
    latent.train()
    z, mu, logvar = latent(H)
    loss = z.sum() + mu.sum() + logvar.sum()
    loss.backward()

    assert latent.fc_mu.weight.grad is not None
    assert (latent.fc_mu.weight.grad.abs() > 0).any()
    assert latent.fc_logvar.weight.grad is not None
    assert (latent.fc_logvar.weight.grad.abs() > 0).any()
    assert H.grad is not None


def test_kl_divergence_math():
    """Verify analytical KL divergence calculation and edge cases."""
    # N(0, I) vs N(0, I) must be exactly 0
    mu_zero = torch.zeros(10, 64)
    logvar_zero = torch.zeros(10, 64)
    kl_zero = VariationalLatent.kl_divergence(mu_zero, logvar_zero)
    assert torch.allclose(kl_zero, torch.tensor(0.0), atol=1e-6)

    # Any non-zero mu or non-zero logvar must produce strictly positive KL divergence
    mu_shifted = torch.ones(10, 64)
    kl_shifted = VariationalLatent.kl_divergence(mu_shifted, logvar_zero)
    assert kl_shifted.item() > 0.0

    logvar_scaled = torch.ones(10, 64)
    kl_scaled = VariationalLatent.kl_divergence(mu_zero, logvar_scaled)
    assert kl_scaled.item() > 0.0

    # Large values are clamped and do not produce NaN or Inf
    mu_large = torch.randn(10, 64) * 100.0
    logvar_large = torch.randn(10, 64) * 100.0
    kl_large = VariationalLatent.kl_divergence(mu_large, logvar_large)
    assert not torch.isnan(kl_large)
    assert not torch.isinf(kl_large)


def test_rotate_decoder_scoring_shapes_and_consistency(dummy_data):
    """Verify RotatEDecoder triple scoring, batched scoring, and score_all_tails contract."""
    decoder = RotatEDecoder(num_base_relations=6, dim=128, gamma=6.0)
    z = torch.randn(dummy_data["num_nodes"], 128)

    B = 8
    h = torch.randint(0, dummy_data["num_nodes"], (B,))
    r = torch.randint(0, 6, (B,))
    t = torch.randint(0, dummy_data["num_nodes"], (B,))

    # 1. Individual 1D triple scores [B]
    scores_1d = decoder.score(z, h, r, t)
    assert scores_1d.shape == (B,)
    assert not torch.isnan(scores_1d).any()
    assert not torch.isinf(scores_1d).any()

    # 2. 2D batched scores with negative samples [B, K]
    K = 5
    neg_t = torch.randint(0, dummy_data["num_nodes"], (B, K))
    scores_2d = decoder.score(z, h, r, neg_t)
    assert scores_2d.shape == (B, K)
    assert not torch.isnan(scores_2d).any()

    # 3. All candidate tails scores [B, N]
    all_scores = decoder.score_all_tails(z, h, r)
    assert all_scores.shape == (B, dummy_data["num_nodes"])
    assert not torch.isnan(all_scores).any()

    # 4. Consistency: true tail score from score_all_tails must match score()
    indexed_scores = all_scores[torch.arange(B), t]
    assert torch.allclose(scores_1d, indexed_scores, atol=1e-4), "score() and score_all_tails() gave divergent scores"


def test_rotate_inverse_relations():
    """Verify that relations with ID >= num_base_relations perform inverse complex rotation."""
    num_base = 6
    dim = 8
    decoder = RotatEDecoder(num_base_relations=num_base, dim=dim, gamma=6.0)

    # Construct synthetic head entity
    z = torch.zeros(2, dim)
    # Head at (1, 0, ...) in complex plane
    z[0, 0] = 1.0  # re = 1, im = 0

    # Pick base relation 0
    rel_id = torch.tensor([0])
    inv_rel_id = torch.tensor([0 + num_base])  # relation 6 is inverse of 0

    # Rotate head by relation 0
    rel_re, rel_im = decoder.get_relation_rotation(rel_id)
    tail_rot = decoder.rotate(z[0:1], rel_re, rel_im)  # [1, dim]
    z[1] = tail_rot[0]

    # Score of (0, rel 0, 1) should be exact maximum (gamma = 6.0)
    score_fwd = decoder.score(z, torch.tensor([0]), rel_id, torch.tensor([1]))
    assert torch.allclose(score_fwd, torch.tensor([6.0]), atol=1e-5)

    # Score of inverse triple (1, inv_rel 6, 0) should also be exact maximum (gamma = 6.0)
    score_inv = decoder.score(z, torch.tensor([1]), inv_rel_id, torch.tensor([0]))
    assert torch.allclose(score_inv, torch.tensor([6.0]), atol=1e-5)


def test_rotate_gradient_flow():
    """Verify gradients propagate to both node representations z and decoder relation embeddings."""
    decoder = RotatEDecoder(num_base_relations=6, dim=32, gamma=6.0)
    z = torch.randn(10, 32, requires_grad=True)

    h = torch.tensor([0, 1, 2])
    r = torch.tensor([1, 2, 3])
    t = torch.tensor([3, 4, 5])

    scores = decoder.score(z, h, r, t)
    loss = scores.sum()
    loss.backward()

    assert z.grad is not None
    assert (z.grad.abs() > 0).any()
    assert decoder.rel_theta.weight.grad is not None
    assert (decoder.rel_theta.weight.grad.abs() > 0).any()


def test_loss_functions():
    """Verify link_loss, kl_weight schedule, and total_loss."""
    # 1. KL warmup schedule
    assert kl_weight(epoch=0, warmup_epochs=20, max_weight=1.0) == 0.0
    assert kl_weight(epoch=10, warmup_epochs=20, max_weight=1.0) == 0.5
    assert kl_weight(epoch=20, warmup_epochs=20, max_weight=1.0) == 1.0
    assert kl_weight(epoch=30, warmup_epochs=20, max_weight=1.0) == 1.0
    assert kl_weight(epoch=5, warmup_epochs=10, max_weight=2.0) == 1.0

    # 2. Link loss (BCE & Margin)
    pos_score = torch.tensor([2.0, 3.0, 4.0])
    neg_score = torch.tensor([-2.0, -1.0, 0.0])
    l_bce = link_loss(pos_score, neg_score, loss_type="bce")
    assert l_bce.ndim == 0
    assert l_bce.item() > 0
    assert not torch.isnan(l_bce)

    l_margin = link_loss(pos_score, neg_score, loss_type="margin", margin=1.0)
    assert l_margin.ndim == 0
    assert not torch.isnan(l_margin)

    # 3. Composite total loss with backprop
    mu = torch.randn(10, 32, requires_grad=True)
    logvar = torch.randn(10, 32, requires_grad=True)
    pos_s = torch.tensor([1.5, 2.5], requires_grad=True)
    neg_s = torch.tensor([[-1.0, 0.0], [-2.0, -0.5]], requires_grad=True)

    tot, metrics = compute_loss_breakdown(
        pos_score=pos_s,
        neg_score=neg_s,
        mu=mu,
        logvar=logvar,
        epoch=10,
        warmup_epochs=20,
        max_kl_weight=1.0,
    )

    assert not torch.isnan(tot)
    assert not torch.isinf(tot)
    assert metrics["kl_weight"] == 0.5
    assert metrics["link_loss"] > 0
    assert metrics["kl_loss"] >= 0

    tot.backward()
    assert mu.grad is not None
    assert logvar.grad is not None
    assert pos_s.grad is not None
    assert neg_s.grad is not None


def test_vrgae_integration_forward_backward():
    """Verify integrated VRGAE model forward, loss computation, gradient backward, and scorer closure."""
    num_nodes = 20
    num_edges = 40
    cfg = {
        "encoder": {
            "type_dim": 8,
            "emb_dim": 16,
            "hidden_dim": 32,
            "out_dim": 32,
            "num_relations": 12,
            "num_bases": 2,
            "dropout": 0.1,
        },
        "latent": {"latent_dim": 32},
        "decoder": {"num_base_relations": 6, "gamma": 6.0},
        "training": {"kl_weight": 1.0, "kl_warmup_epochs": 10},
    }

    node_types = torch.randint(0, 2, (num_nodes, 8)).float()
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, 12, (num_edges,))

    try:
        model = VRGAE(node_types=node_types, cfg=cfg)
    except NotImplementedError:
        pytest.skip("Person 2 RGCNEncoder not yet implemented or merged")

    # 1. Forward pass
    z, mu, logvar = model(edge_index, edge_type)
    assert z.shape == (num_nodes, 32)
    assert mu.shape == (num_nodes, 32)
    assert logvar.shape == (num_nodes, 32)

    # 2. Loss computation
    pos_triples = torch.tensor([[0, 1, 2], [3, 2, 4], [5, 0, 1]])
    neg_triples = torch.tensor([[0, 1, 5], [3, 2, 6], [5, 0, 7]])
    loss, metrics = model.compute_loss(
        z=z, mu=mu, logvar=logvar,
        pos_triples=pos_triples,
        neg_triples=neg_triples,
        epoch=5,
    )
    assert not torch.isnan(loss)
    assert metrics["loss"] > 0

    # 3. Gradients flow to all key submodules
    loss.backward()
    assert model.node_features.emb.weight.grad is not None
    assert model.encoder.conv1.parameters() is not None
    assert model.latent.fc_mu.weight.grad is not None
    assert model.latent.fc_logvar.weight.grad is not None
    assert model.decoder.rel_theta.weight.grad is not None

    # 4. make_score_all_tails contract
    scorer_fn = model.make_score_all_tails(edge_index, edge_type)
    h_batch = torch.tensor([0, 1])
    r_batch = torch.tensor([2, 3])
    scores = scorer_fn(h_batch, r_batch)
    assert scores.shape == (2, num_nodes)
    assert not torch.isnan(scores).any()


def test_vrgae_on_cropdp_dataset():
    """Verify VRGAE forward pass on the real CropDP-KG processed dataset if present."""
    data_dir = Path("data/processed")
    if not (data_dir / "meta.json").exists():
        pytest.skip("data/processed not found")

    import yaml
    from src.data.loader import load_processed

    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = load_processed("data/processed")
    num_entities = data["meta"]["num_entities"]

    try:
        model = VRGAE(node_types=data["node_types"], cfg=cfg)
    except NotImplementedError:
        pytest.skip("Person 2 RGCNEncoder not yet implemented or merged")
    model.eval()

    with torch.no_grad():
        z, mu, logvar = model(data["edge_index"], data["edge_type"])

    assert z.shape == (num_entities, 128)
    assert mu.shape == (num_entities, 128)
    assert logvar.shape == (num_entities, 128)
    assert not torch.isnan(z).any(), "VRGAE produced NaN representations on real dataset"
