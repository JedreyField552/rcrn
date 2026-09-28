# Molecular Graph and Plaque MRI Fusion for Arterial Wall Penetration to Prioritise Carotid Anti-Inflammatory Repurposing

Regime-Coupled Reaction-Norm Fusion (RCRN) scores a candidate anti-inflammatory molecule against one plaque rather than against a model membrane. A graph-and-state-space branch emits a three-route transport-susceptibility vector and a plaque-blind direct effect; a multi-contrast three-dimensional plaque encoder emits a matching resistance vector and a composition index; the pair score is the reaction norm of the two, under a gate that declines to condition when the plaque axis carries no signal. The ranking objective weights each plaque's pairs by the reader-agreement coefficient of its stratum, and the returned top-k list carries an arithmetic certificate or an explicit deferral.

## 1. Installation

With pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

With conda:

```bash
conda env create -f environment.yml
conda activate rcrn
```

With a container:

```bash
docker build -t rcrn .
docker run --rm rcrn
```

The package is not installed in this repository. Every entry point is run either as
`PYTHONPATH=src python3 -m rcrn.cli.<name>` or through the scripts in `scripts/`,
which set `PYTHONPATH` to their own `src/` tree and read `PYTHON_BIN`, defaulting to
`python3`.

## 2. Data

Two layers, kept apart in every report.

**Layer (i) - the clinical outcome layer.** A private, irreversibly anonymised
multi-centre cohort of carotid atherosclerosis patients: three sites across two
regions, multi-contrast high-resolution vessel-wall plaque MRI, plaque composition
labels and cerebrovascular outcome records. It is not redistributed and is not
obtainable from a public repository; it is held under a data-use statement issued
by the corresponding author. No publicly available dataset is used for the primary
clinical outcome.

Because the cohort is not distributed, the release ships a **schema-compatible
cohort fixture** (`rcrn.data.fixture`, `configs/data/fixture.yaml`) that carries the
same record schema and the same relationships the design assumes, so every mechanism
stays executable. The fixture serves execution checks only. It is not a substitute
for the cohort, and it produces no cohort-level value:
**every value in Tables 1 to 4 of the article is `NOT_RUN` in `verification_report.json`,
with the private-cohort reason recorded.** The fixture exposes its own closed form -
route resistances drive the channel intensities, the descriptor block drives the
route susceptibility, and the observed ordering is the reaction norm of the two plus
a recorded observation noise - which is what gives the execution checks a target they
can be wrong about.

To run against the authorised layer instead, set `data.source=cohort` and point
`data.cohort.table`, `data.cohort.candidates`, `data.cohort.volumes` and
`data.cohort.implied_order` at the derived tables. The reader validates the field
contract, refuses any table carrying an identifying column, and never writes a
subject-level record back out.

**Layer (ii) - the public molecular layer.** Auxiliary permeability supervision, the
assay-endpoint comparison, the cross-benchmark check and the pretraining corpus.
Access links are listed in `dataset_urls.txt` together with the version and licence
of each resource, and the attributions are reproduced in `NOTICE`. Only links that
were fetched and whose content matched the named resource are listed; three
resources named by the article could not be verified from the validation
environment and are recorded as unverified rather than listed. The release performs
no network access: the readers take caller-supplied tables, and `scripts/prepare_data.sh`
only checks which of them are present.

Column contracts:

| file | columns |
| --- | --- |
| derived cohort table | `record_id, patient_id, site, region, arm, composition_grade, phenotypes, plaque_volume_mm3, reader_kappa, channels, slice_thickness_mm, motion_grade, vendor, accrual_index, follow_up_days, event_observed, hazard` |
| outcome-implied order | `record_id, candidate_ids` (pipe separated, best first) |
| candidate list | `molecule_id, smiles` |
| assay release | `molecule_id, smiles, target`, optional `cluster` |

## 3. Training

One command per configuration; the configuration name is the experiment name.

```bash
PYTHONPATH=src python3 -m rcrn.cli.train --config configs/experiment/main.yaml
```

Training is two-stage, following the article's end-to-end procedure: the molecular
branch is first supervised on the auxiliary permeability labels so the joint stage
starts from an initialisation rather than from a random one, and the two branches
are then fitted together on the agreement-weighted listwise rank objective. The
interaction covariance, the per-record noise term and the gate threshold are
estimated inside the training partition only.

| configuration | what it changes |
| --- | --- |
| `main.yaml` | the primary result configuration |
| `ablation_no_reaction_norm_coupling.yaml` | Tier 1, the coupling is replaced by plain concatenation |
| `ablation_scalar_susceptibility.yaml` | Tier 1, three routes collapse to one |
| `ablation_no_composition_head.yaml` | Tier 1, no composition index |
| `ablation_no_gate.yaml` | Tier 1, every plaque is conditioned |
| `ablation_uniform_rank_weight.yaml` | Tier 1, uniform pair weights |
| `ablation_empirical_topk.yaml` | Tier 1, no certificate |
| `ablation_imaging_only.yaml` | Tier 1, no molecular branch |
| `ablation_molecular_only.yaml` | Tier 1, no plaque-imaging branch |
| `ablation_joint_*.yaml` | Tier 2, the four joint removals |
| `ablation_bilinear_coupling.yaml` | Tier 3, bilinear concatenation |
| `ablation_two_route.yaml` | Tier 3, a two-route head |
| `ablation_vulnerability_head.yaml` | Tier 3, a vulnerability classifier in place of the composition index |
| `ablation_rank1_covariance.yaml` | Tier 3, the interaction covariance capped at rank one |
| `ablation_full_rank_covariance.yaml` | Tier 3, the covariance left unconstrained |
| `ablation_default_hyperparameters.yaml` | Tier 3, the equal-budget variant |
| `supplementary_*.yaml` | the SI experiments: transport, clinical arms, condition dependence, reader agreement, assay endpoint, constrained parity, gate calibration, mechanistic arm, candidate dispersion |
| `_smoke.yaml` | two-step CPU execution check only; not for reporting |

`--steps N` runs a fixed number of minibatch updates; it exists so the execution
checks and this document's commands are cheap. Do not use it for reporting.

## 4. Evaluation

```bash
PYTHONPATH=src python3 -m rcrn.cli.evaluate  --config configs/experiment/main.yaml --variants
PYTHONPATH=src python3 -m rcrn.cli.certify   --config configs/experiment/main.yaml
PYTHONPATH=src python3 -m rcrn.cli.transport --config configs/experiment/main.yaml
```

`evaluate` writes `evaluation_report.json` and `evaluation_summary.txt` under
`runs/`. It reproduces the mechanism of every reported table on the fixture at the
scale the configuration declares, and labels those values as fixture-scale in the
report's own `scale_note`. It does not substitute fixture-scale numbers for the
article's cohort values.

`certify` runs the certified top-k procedure and reports the certified share, the
deferral rate and the admissible perturbation radius it used. `transport` reports
the paired within-record change and the marginal column difference for each
out-of-distribution fold, because the article's caption states the two can differ.

Reference quantities the article reports, for comparison against an authorised run
and never as run results: hardest-stratum rank agreement 0.69, and an increment of
0.081 over the strongest single-modality baseline with a pre-specified floor of
0.05; pooled incremental area under the curve 0.071 against a published reference of
+0.07, which the prospective arm's 0.064 does not reach and the retrospective arm's
0.073 does; four of eight transport conditions lose more than five percentage
points, the largest being 8.2.

## 5. Verification

```bash
PYTHONPATH=src python3 -m rcrn.cli.verify --config configs/experiment/main.yaml
```

The driver runs two passes and writes four artefacts at the repository root.
`claim_to_code.json` maps every manuscript claim onto the code that carries it and
holds the deviation register. `verification_report.json` records every check as
`PASS`, `FAIL`, `NOT_RUN` or `BLOCKED` from what actually ran, keeps the
manuscript-arithmetic family separate from the code verdict, and reports a
`code_status` beside `overall_status` so a discrepancy inside the article's own
tables is not confused with a broken release. `verification_summary.txt` is the
human-readable form. `integrity_manifest.json` digests the shipped tree and is
written last.

The second pass executes data reading, the forward pass, the loss, the backward
pass, a parameter update, a checkpoint round trip, a minimal training loop and a
single-batch overfit, and recomputes a family of closed forms whose references are
written from the definitions rather than from the implementation: Spearman's
coefficient against the rank-difference formula and an independent library,
Kendall's tau against a brute-force pair count, NDCG against a hand computation,
the rank objective against explicit pair enumeration, the gate statistic and its
covariance against closed forms, the certificate margin against a re-derivation of
the algorithm, Holm and Benjamini-Hochberg against their definitions, the area under
the curve against its pairwise count, a calibration slope against a known
coefficient, and the warm-up cosine factor against its formula.

The third family recomputes relations between the numbers the article itself prints:
cohort partitions, the size-weighted pools of the arm and regional rows, the gate's
complementary shares, the coupling's parameter share, the rounding band of each
printed interaction ratio, every removal's delta against its metric, the transport
threshold count, the assay-error ordering and the rank floor's arithmetic
reachability.

## 6. Compute budget

The article does not report a training budget: no accelerator, no wall-clock, no
storage figure appears in the Methods or the SI. The configuration therefore records
`compute` as not reported and exposes every quantity the article omits as an
engineering default.

An upper-bound estimate from the fixed architecture, to be replaced by the real
figure when the authorised cohort is fitted: 150.7 M parameters at 4 bytes gives
603 MB of weights; AdamW keeps two moments, so optimiser state adds 1.21 GB and the
whole parameter side is about 1.8 GB before activations. With the declared effective
batch of `8 x 1 x 1 = 8` plaque records and 2,486 fitting records, one epoch is 311
steps and the declared 40 epochs are 12,440 steps. Activation memory dominates and
scales with the plaque volume grid, so this estimate is a floor rather than a plan;
the article's own volumes are not distributed and their grid is not printed.

Every executed check in this repository ran on the host CPU. No accelerator was
available, the container runtime is absent, and the private cohort was not
ingested, so those three are reported as `NOT_RUN` or `BLOCKED` rather than
approximated.

## 7. License

The source is distributed under the MIT licence; see `LICENSE`. Third-party
datasets and packages remain under their own terms, listed in `NOTICE` and
`dataset_urls.txt`. Gated weights and the private cohort are never bundled.
