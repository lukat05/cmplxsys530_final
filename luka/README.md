# Luka's Code

This is a list of each of the files and what they do.

### kernel_lib.py

This is the Kernel Library outlined in <b>Task 1.2</b>

### observation.py

This is the Observation / Sampling Process module outlined in <b>Task 2.1</b>.

Key components:
- `sample_cases(tree, pi)` — Uniform i.i.d. sampling of infected hosts with probability π
- `sample_cases_biased(tree, pifn)` — Biased sampling where π(t, x_i) depends on infection time and host covariate
- `build_observed_epidemic_data(...)` — Constructs observed epidemic data Y (incidence, prevalence, metadata)
- `observe(tree, ...)` — End-to-end convenience wrapper combining sampling + Y construction

Run `python3 observation.py` for a built-in self-test.

### summaries.py

This is the Tree and Epidemic Summary Statistics module outlined in <b>Task 2.3</b>.

Key components:
- `compute_tree_summaries(T)` — Computes scalar summary statistics for a sampled transmission genealogy:
  - Branching-time distribution (mean, variance, Q25, Q75)
  - Tree depth (time units and edge count)
  - Cherry count
  - Sackin index (tree imbalance)
  - Colless index (binary trees only)
  - Ladder length (comb-likeness)
- `compute_epidemic_summaries(Y)` — Computes epidemic summary statistics from observed data:
  - Total outbreak size, peak incidence, time to peak, final epidemic size, epidemic duration, mean generation

Run `python3 summaries.py` for a built-in self-test.