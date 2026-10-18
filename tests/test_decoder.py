"""Owner: Person 3. Unit tests for link prediction decoders and VRGAE architecture."""
import pytest
import torch

from src.model.decoder import ComplExDecoder, RotatEDecoder
from src.model.latent import VariationalLatent
from src.model.vrgae import VRGAE


def test_complex_decoder_shapes_and_asymmetry():
    torch.manual_seed(42)
    num_entities = 20
    dim = 64
    num_relations = 6
    batch_size = 8

    decoder = ComplExDecoder(num_base_relations=num_relations, dim=dim)
    z = torch.randn(num_entities, dim)
    h = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    r = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1])
    t = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0])

    scores_forward = decoder.score(z, h, r, t)
    scores_reverse = decoder.score(z, t, r, h)

    assert scores_forward.shape == (batch_size,)
    # ComplEx should break symmetry: forward score != reverse score in general
    assert not torch.allclose(scores_forward, scores_reverse), "ComplEx should be asymmetric"


def test_complex_score_all_tails_consistency():
    torch.manual_seed(42)
    num_entities = 25
    dim = 32
    decoder = ComplExDecoder(num_base_relations=6, dim=dim)
    z = torch.randn(num_entities, dim)
    h = torch.tensor([2, 5])
    r = torch.tensor([1, 4])

    all_scores = decoder.score_all_tails(z, h, r)
    assert all_scores.shape == (2, num_entities)

    # Check that individual candidate scores match row values exactly
    for query_idx in range(2):
        for candidate_t in range(num_entities):
            single_score = decoder.score(
                z,
                h[query_idx:query_idx+1],
                r[query_idx:query_idx+1],
                torch.tensor([candidate_t]),
            )
            assert torch.isclose(all_scores[query_idx, candidate_t], single_score[0], atol=1e-5)


def test_rotate_decoder_shapes_and_values():
    torch.manual_seed(42)
    num_entities = 30
    dim = 64
    num_relations = 6
    gamma = 6.0

    decoder = RotatEDecoder(num_base_relations=num_relations, dim=dim, gamma=gamma)
    z = torch.randn(num_entities, dim)
    h = torch.tensor([0, 1, 2])
    r = torch.tensor([0, 2, 4])
    t = torch.tensor([5, 6, 7])

    scores = decoder.score(z, h, r, t)
    assert scores.shape == (3,)
    assert not torch.isnan(scores).any()

    # Self-triple distance with identity relation should be close to gamma
    # Distance is positive, so score <= gamma
    assert (scores <= gamma + 1e-5).all(), f"RotatE score must not exceed gamma={gamma}"


def test_rotate_score_all_tails_consistency():
    torch.manual_seed(42)
    num_entities = 20
    dim = 32
    decoder = RotatEDecoder(num_base_relations=6, dim=dim, gamma=6.0)
    z = torch.randn(num_entities, dim)
    h = torch.tensor([1, 3])
    r = torch.tensor([2, 4])

    all_scores = decoder.score_all_tails(z, h, r, chunk_size=10)
    assert all_scores.shape == (2, num_entities)

    for q in range(2):
        for candidate_t in range(num_entities):
            single_score = decoder.score(
                z,
                h[q:q+1],
                r[q:q+1],
                torch.tensor([candidate_t]),
            )
            assert torch.isclose(all_scores[q, candidate_t], single_score[0], atol=1e-4)


def test_variational_latent_reparameterization():
    torch.manual_seed(42)
    num_nodes = 40
    in_dim = 64
    latent_dim = 64

    latent = VariationalLatent(in_dim=in_dim, latent_dim=latent_dim)
    H = torch.randn(num_nodes, in_dim)

    # Train mode: stochastic
    latent.train()
    z1, mu1, logvar1 = latent(H)
    z2, mu2, logvar2 = latent(H)
    assert z1.shape == (num_nodes, latent_dim)
    assert not torch.allclose(z1, z2), "Stochastic sampling should produce different z in train mode"
    assert torch.allclose(mu1, mu2), "mu should be deterministic"

    # Eval mode: deterministic (z == mu)
    latent.eval()
    z_eval, mu_eval, _ = latent(H)
    assert torch.allclose(z_eval, mu_eval), "In eval mode z must equal mu"

    # KL divergence test
    kl = VariationalLatent.kl_divergence(mu1, logvar1)
    assert kl.item() >= 0.0, "KL divergence must be non-negative"


def test_vrgae_integration_synthetic():
    torch.manual_seed(42)
    num_nodes = 30
    node_types = torch.randint(0, 2, (num_nodes, 8)).float()
    cfg = {
        "encoder": {
            "emb_dim": 32,
            "hidden_dim": 64,
            "out_dim": 64,
            "num_relations": 12,
            "num_bases": 4,
            "dropout": 0.1,
        },
        "decoder": {
            "num_base_relations": 6,
            "gamma": 6.0,
        },
        "model": {
            "decoder": "rotate",
        },
    }

    src = torch.randint(0, num_nodes, (60,))
    dst = torch.randint(0, num_nodes, (60,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_type = torch.randint(0, 12, (60,))

    vrgae = VRGAE(node_types, cfg)
    z, mu, logvar = vrgae(edge_index, edge_type)
    assert z.shape == (num_nodes, 64)
    assert mu.shape == (num_nodes, 64)
    assert logvar.shape == (num_nodes, 64)

    score_fn = vrgae.make_score_all_tails(edge_index, edge_type)
    h = torch.tensor([0, 1])
    r = torch.tensor([2, 3])
    scores = score_fn(h, r)
    assert scores.shape == (2, num_nodes)
