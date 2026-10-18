"""Owner: Person 2 (Part B). Standalone training of the encoder with a throwaway DistMult scorer.
TODO: load data via src.data.loader.load_processed, build NodeFeatures + RGCNEncoder, train on train triples with
negatives from UniformNegativeSampler (src/data/prepare_cropdp.py), print val MRR on 500 triples.
"""


def main(config_path: str = "configs/default.yaml") -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
