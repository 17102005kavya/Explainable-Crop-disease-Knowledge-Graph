# Team workflow
1. Work on your own branch (`data`, `encoder`, `latent-decoder`, `eval`). Open a merge request into `main`.
2. The next person in the review ring checks it within a few hours: does it run, are the shapes right, is the interface respected.
3. Announce any interface change in the group chat BEFORE making it.
4. Fixed seeds, one config file per run, and every run logged in `results/results.csv`.
5. Daily 15-minute stand-up: done / doing / blocked. Checkpoint at the end of day 4: if the full model is not training, simplify.
