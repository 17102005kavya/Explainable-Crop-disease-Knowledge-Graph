import tempfile
import unittest
from pathlib import Path

from reproduce import compare_metrics, verify_provenance
from src.eval.provenance import DATA_FILES, data_fingerprint, protocol_name


class ProvenanceTests(unittest.TestCase):
    def test_data_fingerprint_changes_when_an_input_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            data_dir = Path(temp)
            for name in DATA_FILES:
                (data_dir / name).write_bytes(name.encode("utf-8"))
            initial = data_fingerprint(data_dir)
            (data_dir / "train.npy").write_bytes(b"updated")
            self.assertNotEqual(initial, data_fingerprint(data_dir))

    def test_data_fingerprint_rejects_incomplete_data(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, "Processed dataset is incomplete"):
                data_fingerprint(Path(temp))

    def test_protocol_names_distinguish_vrgae_training_and_checkpoint_evaluation(self):
        self.assertEqual(
            protocol_name({"model": "vrgae", "source": "train"}),
            "vrgae-shared-training-v2",
        )
        self.assertEqual(
            protocol_name({"model": "vrgae", "source": "checkpoint"}),
            "vrgae-checkpoint-evaluation-v2",
        )

    def test_protocol_names_identify_no_graph_ablation(self):
        config = {"model": "rgcn_distmult", "params": {"use_graph": False}}
        self.assertEqual(protocol_name(config), "distmult-no-graph-shared-runner-v2")

    def test_legacy_rows_are_not_claimed_as_reproducible(self):
        with self.assertRaisesRegex(ValueError, "legacy/unverified"):
            verify_provenance({"overall": {"provenance_status": "legacy-unverified"}})

    def test_metric_comparison_enforces_tolerance_and_selected_epoch(self):
        expected = {"overall": {"selected_epoch": "10", "MRR": "0.5000",
                                "Hits@1": "0.4", "Hits@3": "0.6",
                                "Hits@10": "0.8", "AUC": "0.7"}}
        actual = {"overall": dict(expected["overall"], MRR="0.5040")}
        self.assertEqual(compare_metrics(expected, actual, 0.005), [])
        actual["overall"]["MRR"] = "0.5060"
        self.assertTrue(compare_metrics(expected, actual, 0.005))


if __name__ == "__main__":
    unittest.main()
