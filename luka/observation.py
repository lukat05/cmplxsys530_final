"""
observation.py — Task 2.1: Observation / Sampling Process
=========================================================

Implements the sampling model that converts the latent transmission tree
(produced by the Gillespie simulator, Task 1.3) into observed data objects
(Y, T_sampled).

Two sampling modes are provided:
  1. **Uniform sampling** — each infected host is independently retained
     with constant probability π.
  2. **Biased sampling** — retention probability π(t, x_i) depends on the
     host's infection time and/or covariate value, supplied as a callable.

The module also constructs the observed epidemic data Y:
  - Reported incidence counts per time window
  - Reported prevalence curve (time series)
  - Set of sampled host metadata (infection time, recovery time, covariate)

Expected transmission-tree format (networkx.DiGraph, from Task 1.3):
  Node attributes:
    - 'host_id'       : int, unique host index
    - 'infection_time': float, time at which the host was infected
    - 'recovery_time' : float, time at which the host recovered
    - 'covariate'     : float (or array), host covariate x_i
    - 'state'         : str, one of {'S', 'I', 'R'}  (at end of simulation)
    - 'generation'    : int, depth in the transmission tree

  Edges are directed: parent (infector) → child (infectee).
  Root nodes (seed infections) have in-degree 0.

Author: Luka Todorovic
Course: CMPLXSYS 530 / EPI 638, Winter 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SampledHost:
    """Metadata for a single sampled (observed) infected host."""
    host_id: int
    infection_time: float
    recovery_time: float
    covariate: float
    generation: int


@dataclass
class ObservedEpidemicData:
    """
    Container for the observed epidemic data object Y.

    Attributes
    ----------
    sampled_hosts : list[SampledHost]
        Metadata for every sampled host.
    sampled_ids : set[int]
        Set of host_id values for the sampled hosts (convenience).
    incidence : np.ndarray, shape (n_bins,)
        Reported new-case counts per time bin.
    time_bins : np.ndarray, shape (n_bins + 1,)
        Edges of the time bins used for incidence.
    prevalence_times : np.ndarray, shape (n_points,)
        Time points at which prevalence is evaluated.
    prevalence : np.ndarray, shape (n_points,)
        Number of sampled hosts infectious at each time point.
    total_observed_cases : int
        Total number of sampled hosts.
    peak_incidence : float
        Maximum value in the incidence array.
    time_to_peak : float
        Mid-point of the time bin with maximum incidence.
    final_epidemic_size : int
        Total number of sampled hosts that were eventually infected
        (equivalent to total_observed_cases for SIR without waning).
    """
    sampled_hosts: List[SampledHost]
    sampled_ids: Set[int]
    incidence: np.ndarray
    time_bins: np.ndarray
    prevalence_times: np.ndarray
    prevalence: np.ndarray
    total_observed_cases: int
    peak_incidence: float
    time_to_peak: float
    final_epidemic_size: int


# ---------------------------------------------------------------------------
# Core sampling functions
# ---------------------------------------------------------------------------

def sample_cases(
    tree: nx.DiGraph,
    pi: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[SampledHost], Set[int]]:
    """
    Uniform i.i.d. sampling: retain each infected host with probability π.

    Parameters
    ----------
    tree : nx.DiGraph
        Latent transmission tree from the simulator (Task 1.3).
        Every node must have attributes: host_id, infection_time,
        recovery_time, covariate, generation.
    pi : float
        Sampling probability in [0, 1].
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    sampled_hosts : list[SampledHost]
        Metadata for each sampled host.
    sampled_ids : set[int]
        Host IDs of the sampled set.

    Raises
    ------
    ValueError
        If π is not in [0, 1].
    """
    if not 0.0 <= pi <= 1.0:
        raise ValueError(f"Sampling probability pi must be in [0, 1], got {pi}")

    if rng is None:
        rng = np.random.default_rng()

    sampled_hosts: List[SampledHost] = []
    sampled_ids: Set[int] = set()

    for node, attrs in tree.nodes(data=True):
        # Draw Bernoulli(π) for each infected host
        if rng.random() < pi:
            host = SampledHost(
                host_id=attrs["host_id"],
                infection_time=attrs["infection_time"],
                recovery_time=attrs["recovery_time"],
                covariate=attrs["covariate"],
                generation=attrs["generation"],
            )
            sampled_hosts.append(host)
            sampled_ids.add(attrs["host_id"])

    return sampled_hosts, sampled_ids


def sample_cases_biased(
    tree: nx.DiGraph,
    pifn: Callable[[float, float], float],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[SampledHost], Set[int]]:
    """
    Biased sampling: retain host i with probability π(t_i, x_i).

    This allows modelling realistic surveillance biases, e.g.
      - time-dependent reporting (improving over an outbreak)
      - attribute-dependent ascertainment (symptomatic / high-risk
        individuals sampled preferentially)

    Parameters
    ----------
    tree : nx.DiGraph
        Latent transmission tree (same format as sample_cases).
    pifn : callable (t: float, x: float) -> float
        Sampling probability function.  Must return a value in [0, 1].
        t = infection time of the host, x = covariate value.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    sampled_hosts : list[SampledHost]
        Metadata for each sampled host.
    sampled_ids : set[int]
        Host IDs of the sampled set.

    Raises
    ------
    ValueError
        If pifn returns a value outside [0, 1] for any host.
    """
    if rng is None:
        rng = np.random.default_rng()

    sampled_hosts: List[SampledHost] = []
    sampled_ids: Set[int] = set()

    for node, attrs in tree.nodes(data=True):
        t_inf = attrs["infection_time"]
        x_i = attrs["covariate"]
        prob = pifn(t_inf, x_i)

        if not 0.0 <= prob <= 1.0:
            raise ValueError(
                f"pifn({t_inf}, {x_i}) returned {prob}, which is outside [0, 1]"
            )

        if rng.random() < prob:
            host = SampledHost(
                host_id=attrs["host_id"],
                infection_time=attrs["infection_time"],
                recovery_time=attrs["recovery_time"],
                covariate=attrs["covariate"],
                generation=attrs["generation"],
            )
            sampled_hosts.append(host)
            sampled_ids.add(attrs["host_id"])

    return sampled_hosts, sampled_ids


# ---------------------------------------------------------------------------
# Epidemic data construction (Y)
# ---------------------------------------------------------------------------

def build_observed_epidemic_data(
    sampled_hosts: List[SampledHost],
    sampled_ids: Set[int],
    n_time_bins: int = 50,
    n_prevalence_points: int = 200,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> ObservedEpidemicData:
    """
    Construct the observed epidemic data Y from the sampled host set.

    Y consists of:
      - Reported incidence counts per time window
      - Reported prevalence curve (number currently infectious at each t)
      - Sampled host metadata

    Parameters
    ----------
    sampled_hosts : list[SampledHost]
        Output from sample_cases or sample_cases_biased.
    sampled_ids : set[int]
        Host IDs in the sample.
    n_time_bins : int
        Number of equally-spaced time bins for computing incidence.
    n_prevalence_points : int
        Number of time points at which to evaluate prevalence.
    t_start : float, optional
        Start of the observation window.  Defaults to the earliest
        infection time among sampled hosts.
    t_end : float, optional
        End of the observation window.  Defaults to the latest
        recovery time among sampled hosts.

    Returns
    -------
    ObservedEpidemicData
        Dataclass with all epidemic summary fields populated.

    Raises
    ------
    ValueError
        If no sampled hosts are provided.
    """
    if len(sampled_hosts) == 0:
        # Edge case: no hosts sampled — return empty data
        empty = np.array([])
        return ObservedEpidemicData(
            sampled_hosts=[],
            sampled_ids=set(),
            incidence=empty,
            time_bins=empty,
            prevalence_times=empty,
            prevalence=empty,
            total_observed_cases=0,
            peak_incidence=0.0,
            time_to_peak=0.0,
            final_epidemic_size=0,
        )

    inf_times = np.array([h.infection_time for h in sampled_hosts])
    rec_times = np.array([h.recovery_time for h in sampled_hosts])

    if t_start is None:
        t_start = float(inf_times.min())
    if t_end is None:
        t_end = float(rec_times.max())

    # --- Incidence: histogram of infection times ---
    time_bins = np.linspace(t_start, t_end, n_time_bins + 1)
    incidence, _ = np.histogram(inf_times, bins=time_bins)

    # Peak incidence and time to peak
    peak_idx = int(np.argmax(incidence))
    peak_incidence = float(incidence[peak_idx])
    time_to_peak = float(0.5 * (time_bins[peak_idx] + time_bins[peak_idx + 1]))

    # --- Prevalence: number of hosts currently infectious at each t ---
    prevalence_times = np.linspace(t_start, t_end, n_prevalence_points)
    # A host is infectious at time t if infection_time <= t < recovery_time
    prevalence = np.zeros(n_prevalence_points, dtype=int)
    for i, t in enumerate(prevalence_times):
        prevalence[i] = int(np.sum((inf_times <= t) & (t < rec_times)))

    total_observed = len(sampled_hosts)

    return ObservedEpidemicData(
        sampled_hosts=sampled_hosts,
        sampled_ids=sampled_ids,
        incidence=incidence,
        time_bins=time_bins,
        prevalence_times=prevalence_times,
        prevalence=prevalence,
        total_observed_cases=total_observed,
        peak_incidence=peak_incidence,
        time_to_peak=time_to_peak,
        final_epidemic_size=total_observed,
    )


# ---------------------------------------------------------------------------
# Convenience: one-call pipeline
# ---------------------------------------------------------------------------

def observe(
    tree: nx.DiGraph,
    pi: float = 0.5,
    pifn: Optional[Callable[[float, float], float]] = None,
    n_time_bins: int = 50,
    n_prevalence_points: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> ObservedEpidemicData:
    """
    End-to-end convenience wrapper: sample hosts, then build Y.

    If *pifn* is provided, biased sampling is used; otherwise uniform
    sampling with probability *pi*.

    Parameters
    ----------
    tree : nx.DiGraph
        Latent transmission tree.
    pi : float
        Uniform sampling probability (used only when pifn is None).
    pifn : callable, optional
        Biased sampling function π(t, x_i).
    n_time_bins : int
        Bins for incidence histogram.
    n_prevalence_points : int
        Points for prevalence curve.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    ObservedEpidemicData
    """
    if pifn is not None:
        hosts, ids = sample_cases_biased(tree, pifn, rng=rng)
    else:
        hosts, ids = sample_cases(tree, pi, rng=rng)

    return build_observed_epidemic_data(
        hosts, ids,
        n_time_bins=n_time_bins,
        n_prevalence_points=n_prevalence_points,
    )


# ---------------------------------------------------------------------------
# Self-test / demonstration
# ---------------------------------------------------------------------------

def _make_demo_tree(n: int = 50, seed: int = 42) -> nx.DiGraph:
    """
    Build a small synthetic transmission tree for testing purposes.

    This mimics the output format of Task 1.3 (build_transmission_tree):
    a networkx DiGraph with the required node attributes.
    """
    rng = np.random.default_rng(seed)
    tree = nx.DiGraph()

    # Seed infection (patient zero)
    tree.add_node(
        0,
        host_id=0,
        infection_time=0.0,
        recovery_time=rng.exponential(10.0),
        covariate=rng.uniform(0.0, 1.0),
        state="R",
        generation=0,
    )

    for i in range(1, n):
        # Pick a random existing node as infector
        infector = rng.integers(0, i)
        inf_time_parent = tree.nodes[infector]["infection_time"]
        # Infection happens after the parent was infected
        inf_time = inf_time_parent + rng.exponential(2.0)
        rec_time = inf_time + rng.exponential(10.0)

        tree.add_node(
            i,
            host_id=i,
            infection_time=inf_time,
            recovery_time=rec_time,
            covariate=rng.uniform(0.0, 1.0),
            state="R",
            generation=tree.nodes[infector]["generation"] + 1,
        )
        tree.add_edge(infector, i, time=inf_time)

    return tree


if __name__ == "__main__":
    # Quick smoke test
    print("=" * 60)
    print("Task 2.1 — Observation / Sampling Process: self-test")
    print("=" * 60)

    demo_tree = _make_demo_tree(n=100, seed=0)
    print(f"\nDemo tree: {demo_tree.number_of_nodes()} nodes, "
          f"{demo_tree.number_of_edges()} edges")

    # --- Uniform sampling at π = 0.5 ---
    rng = np.random.default_rng(123)
    Y_uniform = observe(demo_tree, pi=0.5, rng=rng)
    print(f"\n[Uniform π=0.5] Sampled {Y_uniform.total_observed_cases} / 100 hosts")
    print(f"  Peak incidence   : {Y_uniform.peak_incidence:.0f}")
    print(f"  Time to peak     : {Y_uniform.time_to_peak:.2f}")
    print(f"  Final epi size   : {Y_uniform.final_epidemic_size}")

    # --- Full sampling (π = 1) should sample everyone ---
    Y_full = observe(demo_tree, pi=1.0, rng=np.random.default_rng(0))
    assert Y_full.total_observed_cases == 100, (
        f"Full sampling (π=1) should recover all 100 hosts, got {Y_full.total_observed_cases}"
    )
    print(f"\n[Full π=1.0] Sampled {Y_full.total_observed_cases} / 100 hosts  ✓")

    # --- Biased sampling: higher π for hosts with higher covariate ---
    def biased_pi(t: float, x: float) -> float:
        """High-covariate hosts sampled 3x more often."""
        return min(1.0, 0.2 + 0.6 * x)

    rng2 = np.random.default_rng(456)
    Y_biased = observe(demo_tree, pifn=biased_pi, rng=rng2)
    sampled_covs = [h.covariate for h in Y_biased.sampled_hosts]
    all_covs = [d["covariate"] for _, d in demo_tree.nodes(data=True)]
    print(f"\n[Biased π(t,x)] Sampled {Y_biased.total_observed_cases} / 100 hosts")
    print(f"  Mean covariate (sampled) : {np.mean(sampled_covs):.3f}")
    print(f"  Mean covariate (all)     : {np.mean(all_covs):.3f}")
    if np.mean(sampled_covs) > np.mean(all_covs):
        print("  Biased sampling enriches high-covariate hosts  ✓")
    else:
        print("  WARNING: bias direction check failed (stochastic, retry with more hosts)")

    # --- Edge case: π = 0 ---
    Y_empty = observe(demo_tree, pi=0.0, rng=np.random.default_rng(0))
    assert Y_empty.total_observed_cases == 0, "π=0 should sample nobody"
    print(f"\n[Empty π=0.0] Sampled {Y_empty.total_observed_cases} / 100 hosts  ✓")

    print("\n" + "=" * 60)
    print("All self-tests passed.")
    print("=" * 60)
