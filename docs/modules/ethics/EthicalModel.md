# EthicalModel

`EthicalModel` is a sub-model of the BDT mobility digital twin. It measures the impact of a policy on vulnerable population segments.

---

## Architecture overview

`EthicalModel` is built as a `civic_digital_twins.dt_model.Model` sub-model. These three dataclasses define the model's contract with the outside world:

```
EthicalModel(Model)
├── Inputs: symbolic parameters fed in by the parent model
├── Outputs: symbolic results consumed by the parent model
└── Expose: internal quantities made visible to the parent model
```

All values in `Inputs`, `Outputs`, and `Expose` are `Index` objects.
> **No concrete numbers are produced inside the model.**

Concrete float values are computed at simulation time when the parent model calls `evaluate()` and produces a `subs` OR for plotting purposes only.
this is done so that the  model can be wired into an ensemble simulation without code refactoring.

---

## Data flow

The pipeline has **static knowledge inputs** (loaded once from disk) and **dynamic policy inputs** (supplied by parent model as `Index` objects at construction time).

**1. Load static knowledge** — `CaseLoader` reads parquets, normalises geo-indicators (with minors-specific shared min/max for parent columns), then everything is converted into nested `Index` dicts: normalised indicators, MPI weights, vulnerability weights, and sensibility weights.

**2. Compute baseline** → from static indicators, two quantities are derived: `vulnerability_by_area` and the baseline MPI map.

**3. Apply Policy** → `policy.apply()` takes baseline indicators, vulnerability scores, sensibility weights, and the caller-supplied policy parameters: returns `modified_by_area` (identical shape where each value has been modified).

**4. Compute modified world** → the modified and delta MPI maps are derived from `modified_by_area` using the same MPI aggregation in step 2.

**5. Outputs** → results are projected to the requested `year` and stored in `Outputs` and `Expose`.

---

## Public API

The package exports the model API plus projection helpers from its root `__init__.py`:

```python
from bdt_mobility.modules.ethics.model import (
    EthicalModelBundle,
    EthicalModel,
    EthicalCaseSpec,
    ResolvedEthicalSliceSpec,
    GenderKey,
    CASE_SPECS,
    resolve_case_slice,
    get_case_spec,
    Policy,
    CareTimePolicy,
    EconomicBurdenPolicy,
    CaseLoader,
    ProjectionEntry,
    ProjectionMap,
    build_projection_map_from_zone_ids,
    project_float_map,
    project_index_map,
)
```

### `EthicalModelBundle`

Preferred entry point. Orchestrates one `EthicalModel` child for each gender by sharing the same policy inputs across them:

```python
from civic_digital_twins.dt_model import Index
from bdt_mobility.modules.ethics.model import EthicalModelBundle

bundle = EthicalModelBundle(
    ztl_start=Index("ztl_start", 7.0),
    ztl_end=Index("ztl_end", 20.0),
    avail_postpone=Index("avail_postpone", 0.3),
    avail_anticipate=Index("avail_anticipate", 0.2),
    median_postpone=Index("median_postpone", 1.5),
    median_anticipate=Index("median_anticipate", 1.0),
    case_id="elderly",
    genders=("None", "male", "female"),
    projection=projection_map,
    k_scale=1.0,
)
```

Care-time params (`avail_*`, `median_*`) are optional — only needed by `CareTimePolicy` cases.
For the `income` case pass `ticket_costs` instead.

Bundle now exposes keyed maps:

| Attribute | Type | Description |
|---|---|---|
| `models["None"]` | `EthicalModel` | Aggregate slice |
| `models["male"]` | `EthicalModel` | Male slice |
| `models["female"]` | `EthicalModel` | Female slice |
| `outputs.baseline_mpi[gender]` | `dict[int, Index]` | Pre-policy MPI for one slice |
| `outputs.modified_mpi[gender]` | `dict[int, Index]` | Post-policy MPI for one slice |
| `outputs.delta_mpi[gender]` | `dict[int, Index]` | Signed difference for one slice |

### `EthicalModel`

Internal single-entry engine (still reachable for a single `(case, gender)`):

```python
from civic_digital_twins.dt_model import Index
from bdt_mobility.modules.ethics.model import EthicalModel

model = EthicalModel(
    ztl_start=Index("ztl_start", 7.0),        # hour ZTL opens
    ztl_end=Index("ztl_end", 20.0),           # hour ZTL closes
    avail_postpone=Index("avail_postpone", 0.3),    # optional, for care-time cases
    avail_anticipate=Index("avail_anticipate", 0.2),
    median_postpone=Index("median_postpone", 1.5),
    median_anticipate=Index("median_anticipate", 1.0),
    case_id="elderly",
    gender="None",
    year=2023,
    projection=projection_map,
    k_scale=1.0,
)
```

Optional constructor params: `avail_postpone`, `avail_anticipate`, `median_postpone`, `median_anticipate` (for care-time cases), `ticket_costs: dict[str, Index]` (for income case), `base_dir: Path` (override data root), `k_scale: float` (policy intensity multiplier, default 1.0).

The model exposes:

| Attribute | Type | Description |
|---|---|---|
| `outputs.baseline_mpi` | `dict[int, Index]` | Pre-policy MPI |
| `outputs.modified_mpi` | `dict[int, Index]` | Post-policy MPI |
| `outputs.delta_mpi` | `dict[int, Index]` | Signed difference (modified - baseline) |
| `expose.vulnerability_by_area` | `dict[int, Index]` | Normalised vulnerability score |

`output_geography` tells you how to interpret the MPI map keys:
- `"area_statistica"` means keys are source statistical-area ids
- `"zone"` means keys are simulation zone indices

When a `projection` is supplied, the three MPI outputs are projected to zone space.
`expose.vulnerability_by_area` stays keyed by statistical area since that is the correct underlying behaviour.

Also, these available indexes can be accessed for per-indicator inspection:
`I_baseline_mpi`, `I_modified_mpi`, `I_delta_mpi`, `I_vulnerability_by_area`

`slice_id` exposes the case/gender ("slice") identity, e.g. `"elderly/None"` or `"elderly/female"`.


### `EthicalCaseSpec` / `ResolvedEthicalSliceSpec`

`EthicalCaseSpec` keeps case-only configuration: sector names, normalization settings, survey columns, inverse columns, and the `Policy` instance.

`ResolvedEthicalSliceSpec` is derived from `(case_id, gender)` and contains the filenames, schemas, MPI column names, and output labels for one slice.

`resolve_case_slice(case_id, gender)` is the resolver that links the two levels.

Three cases are currently in `CASE_SPECS`:

| `case_id` | Population segment | Policy | Care intervals |
|---|---|---|---|
| `"elderly"` | Elderly + caregivers | `CareTimePolicy` | none |
| `"minors"` | Families with minors | `CareTimePolicy` | `(7–9h, 12–13h, 15–16h)` = school pickup/drop-off windows |
| `"income"` | Families income-sensitive | `EconomicBurdenPolicy` | n/a (ticket-based) |

Adding a new case-study means adding one `EthicalCaseSpec` entry to `CASE_SPECS` and supplying the matching `Policy` instance. If the new case needs special column handling during normalisation, add a branch in `CaseLoader._prepare_indicators()`.

---

## Package structure

```
model/
├── __init__.py
├── bundle.py
├── model.py
├── cases.py
├── policy.py
├── projection.py
├── core/
│   ├── __init__.py
│   ├── accept_zone.py
│   ├── builders.py
│   ├── burden.py
│   ├── mpi.py
│   ├── normalization.py
│   ├── ticket_indicators.py
│   └── vulnerability.py
├── loading/
│   ├── __init__.py
│   ├── loaders.py
│   └── schemas.py
└── viz/
    ├── __init__.py
    └── plotting.py
```

---

## File-by-file description

### `model.py`

Orchestrates the pipeline inside `__init__`:

1. input `Index` objects into `EthicalModel.Inputs`.
2. calls `CaseLoader.load()` to read parquet files and produce nested `Index` dicts for indicators, weights, and MPI.
3. computes `vulnerability_by_area`.
4. computes the baseline MPI structure.
5. builds `policy_params` (converts seconds to hours, normalises availability), filtering out `None` params.
6. calls `case_spec.policy.apply()`.
7. computes the modified MPI structure and delta maps from the modified indicators.
8. selects the requested `year`.
9. optionally projects MPI outputs to the requested output geography.
10. fills `Outputs` and `Expose`.
11. `super().__init__()`

`extract_output_maps(subs)`: returns floats for `baseline_mpi`, `modified_mpi`, and `delta_mpi`.


### `bundle.py`

Orchestrator over one or more `EthicalModel` children.

- constructs one child model for each gender
- exposes keyed outputs `"None"`, `"male"`, `"female"`
- collects all child `indexes` into a flat list for registration in the parent model
- `extract_output_maps(subs)` packages all child outputs into a dict keyed by gender


### `cases.py`

Defines the `CASE_SPECS` options, `GenderKey` type alias, and the resolver for one slice.
Adding a new case-study means adding one base spec here and reusing the same gender-resolution pattern.

### `policy.py`

File that translates the policy parameters into a modification of the indicators.

**`Policy`** -- abstract Policy class:
- `required_params()`: returns the parameters the policy needs from `EthicalModel.Inputs`.
- `apply()`: produces `modified_by_area` (actual model results).
- `validate()`: checks that all required params are present.

**`CareTimePolicy`** -- for `elderly` and `minors`

1. computes overlap of the ZTL window with each care interval and derives `overlap_factor`.
2. combines `overlap_factor` with caregiver flexibility and median displacement hours to produce `stress_factor`.
3. for each (area, year, indicator): `modified = baseline * (1 + k_scale * stress * sensitivity * vulnerability)`.

Result is a dict of the same shape as the baseline indicators.

When `care_intervals` is empty (elderly case) the overlap factor falls back to a fixed constant (`overlap_fallback`).

**`EconomicBurdenPolicy`** -- for `income`

1. delegates to `core/burden.py` to compute a per-area ticket effect from accept zones, ticket prices, and euro-class splits.
2. delegates to `core/ticket_indicators.py` to apply the ticket effect to each indicator: `modified = baseline * (1 + k_scale * ticket_effect * sensitivity)`.
3. requires `accept_zone_by_area` (computed by `CaseLoader` from income data).
4. required params: `ticket_costs` (dict of euro-class ticket prices as `Index` objects).

`_sym_min` and `_sym_max` are local helpers because `graph` provides `graph.maximum` but no `graph.minimum` (from my research, could be present tho)

### `projection.py`

Module to map statistical-area outputs onto simulation zones.

- `build_projection_map_from_zone_ids`: builds the mapping from crosswalk + simulation zones.
- `project_float_map`: projects numerical outputs.
- `project_index_map`: projects symbolic `Index` outputs.


### `loading/loaders.py`

`CaseLoader` is the data loading orchestrator:

1. reads the case-specific indicator parquet.
2. loads population data.
3. for income case: computes `accept_zone_by_area` and `exempt_share_by_area`.
4. calls `_prepare_indicators()` to select, canonicalize, and normalize columns.
5. loads support data (MPI weights, vulnerability weights, sensibility weights).
6. calls `_build_index_from_schema` to convert dataframes into nested `Index` dicts.
7. returns `EthicalCaseInputs` dataclass packaging everything.

`_prepare_indicators`: inline normalization pipeline. For elderly/income: selects columns, computes reference values, normalizes. For minors: additionally remaps gender-specific parent columns (e.g. `perc_single_female_family_14y` → `perc_single_parents_14y`), then normalizes parent columns separately with shared min/max across gender variants.

`_resolve_base_dir` resolves the repository root from `__file__` if no explicit path is passed (to make model self-contained).

`_build_index_from_schema`: generic converter driven by schema dicts from `schemas.py`. Melts value columns, iterates rows, builds nested dicts with `Index` leaves.

`_reshape_to_area_first`: pivots `{indicator -> year -> area -> Index}` into `{area -> year -> indicator -> Index}`.

### `loading/schemas.py`

`INDEX_SCHEMAS` dict: one entry per parquet/data source. Each schema defines `key_cols`, `value_cols` (or `value_col`), `name_prefix`, optionally `nested_key_order`, `canonical_value_cols`, and `column_roles`.

Schemas exist for: MPI weights, vulnerability/sensibility weights, and all indicator variants (elderly/minors/economic x None/male/female). The minors schemas include `column_roles` mappings for gender-specific parent columns.

### `core/mpi.py`

`compute_mpi_index_by_area`: MPI `Index` for one (area, year) as weighted sum of indicators.

### `core/builders.py`

Helpers to iterate over `{area -> {year -> {indicator -> Index}}}` nested dict structure:

- `compute_mpi_by_area`: calls `core/mpi.py` for each (area, year).
- `compute_mpi_deltas`: baseline MPI minus modified MPI for each (area, year).
- `compute_indicator_deltas`: per-indicator deltas (currently unused but available).

### `core/normalization.py`

Normalization pipeline applied at data load time (can be saved permanently in future revisions):

- `load_population_by_area`: loads resident counts per (area, year) from .parquet.
- `compute_reference_values`: population-weighted mean for each indicator for the reference "neutral" baseline for normalization.
- `normalize_indicators`: `100 + 100 * (value - ref) / (max - min)` across the 2021-2023 range, survey-only columns are excluded. Inverse columns use `100 - 100 * (value - ref) / denom`.

> **A second normalization can be made to center around 100.**
Right now this second normalization is omitted, can be added any time.

### `core/vulnerability.py`

`compute_vulnerability_by_area`: per-area vulnerability score as weighted sum of indicators, normalised against the year's mean. Result is a dict of `Index`.

### `core/accept_zone.py`

`compute_accept_zone`: maps per-area income into a "ticket acceptance threshold" clamped to `[global_min, global_max]` (default 4-7). Used exclusively by the income case. Formula: `midpoint * (area_income / ref_income)`.

### `core/burden.py`

`compute_ticket_effect`: for each area, computes a weighted burden index across euro classes. Per class: `(ticket_price / accept_zone - 1)` clipped to `[0, 1]`, then weighted by euro-class population split. Optionally discounts by exempt share. Returns `dict[area, Index]`.

### `core/ticket_indicators.py`

`apply_ticket_to_indicators`: applies the ticket effect from `burden.py` to all indicators: `modified = baseline * (1 + k_scale * ticket_effect * sensitivity)`. Indicators without a sensibility weight are passed through unchanged.

### `viz/plotting.py`

`plot_pums_zones_choropleth`: generic choropleth renderer for any GeoDataFrame with a value column. Configurable color scale, hover data, and color range.

`plot_mpi_map`: wrapper that takes simulation-zone keyed MPI values (`dict[int, float]`), attaches them to the OD zones GeoDataFrame, validates zone keys, and calls `plot_pums_zones_choropleth`.

Zone geometry `od_zones` is to be `aree_gdf_inside` as in `MobilityModel.default_config` (`bdt_mobility/model/model.py`) (OD zones intersected with the ZOI boundary). `MobilityModel` builds the `EthicalModel` bundles only when `enable_ethics` is set in `config_params`.
