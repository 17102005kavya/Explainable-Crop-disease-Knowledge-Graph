"""Owner: Person 3 (Part C). Training loop for the full model.  TODO: implement."""


def main(config_path: str = "configs/default.yaml") -> None:
    # TODO: load data -> VRGAE -> loop (positives from train.npy, negatives from UniformNegativeSampler,
    #       total_loss with KL warm-up) -> evaluate every cfg["training"]["eval_every"] epochs -> save best checkpoint
    raise NotImplementedError


if __name__ == "__main__":
    main()
