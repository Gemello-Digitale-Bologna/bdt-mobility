from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from civic_digital_twins.dt_model import ConstIndex

from bdt_mobility.utils.paths import PathRegistry

from .schemas import INDEX_SCHEMAS
from ..cases import ResolvedEthicalSliceSpec
from ..core.normalization import compute_reference_values, load_population_by_area, normalize_indicators


@dataclass(frozen=True)
class EthicalCaseInputs:
    dfs: dict
    indicators: dict
    vulnerability_weights: dict
    sensibility_weights: dict
    mpi_weights: dict
    accept_zone_by_area: dict[int, float] | None = None
    exempt_share_by_area: dict[int, float] | None = None


class CaseLoader:
    def __init__(self, slice_spec: ResolvedEthicalSliceSpec, *, base_dir: Path | None = None) -> None:
        self.slice_spec = slice_spec
        self.base_dir = self._resolve_base_dir(base_dir)
        self.analysis_outputs = PathRegistry.ANALYSIS_OUTPUTS

    def load(self) -> EthicalCaseInputs:
        slice_spec = self.slice_spec
        case_spec = slice_spec.case_spec
        schema = INDEX_SCHEMAS[slice_spec.indicators_schema]
        indicator_names = set(schema.get("canonical_value_cols", schema["value_cols"]))

        df_raw = self._load_raw_indicators()
        df_pop = self._load_population()

        accept_zone_by_area = None
        exempt_share_by_area = None
        if case_spec.case_id == "income":
            accept_zone_by_area = self._compute_accept_zone(df_raw, df_pop)
            exempt_share_by_area = self._compute_exempt_share(df_raw)

        df_indicators = self._prepare_indicators(df_raw, df_pop, schema)

        dfs = self._load_support_data(df_indicators)

        indexed_vulnerability_weights = self._build_index_from_schema(
            dfs["vulnerability_weights"], INDEX_SCHEMAS["vulnerabilityWeights"],
        )
        indexed_sensibility_weights = self._build_index_from_schema(
            dfs["sensibility_weights"], INDEX_SCHEMAS["sensibilityWeights"],
        )
        indexed_indicators = self._reshape_to_area_first(
            self._build_index_from_schema(df_indicators, schema)
        )
        indexed_mpi_weights = self._build_index_from_schema(
            dfs["MPI_weights"], INDEX_SCHEMAS["MPI_weights"],
        )

        return EthicalCaseInputs(
            dfs=dfs,
            indicators=indexed_indicators,
            vulnerability_weights={
                key: value
                for key, value in indexed_vulnerability_weights.get(case_spec.sector_key, {}).get(slice_spec.gender, {}).items()
                if key in indicator_names
            },
            sensibility_weights={
                key: value
                for key, value in indexed_sensibility_weights[case_spec.sector_key].items()
                if key in indicator_names
            },
            mpi_weights={
                key: value
                for key, value in indexed_mpi_weights[case_spec.mpi_sector_key].items()
                if key in indicator_names
            },
            accept_zone_by_area=accept_zone_by_area,
            exempt_share_by_area=exempt_share_by_area,
        )

    def _load_raw_indicators(self) -> pd.DataFrame:
        return pd.read_parquet(self.slice_spec.indicators_path).fillna(0)

    def _load_population(self) -> pd.DataFrame:
        return load_population_by_area()

    def _prepare_indicators(self, df_raw: pd.DataFrame, df_pop: pd.DataFrame, schema: dict) -> pd.DataFrame:
        slice_spec = self.slice_spec
        case_spec = slice_spec.case_spec
        canonical_cols = list(schema.get("canonical_value_cols", schema["value_cols"]))
        geo_cols = [c for c in canonical_cols if c not in case_spec.survey_cols]

        if case_spec.case_id != "minors":
            df = df_raw[schema["key_cols"] + list(schema["value_cols"])].copy()
            ref = compute_reference_values(df, df_pop, geo_cols, ref_year=case_spec.ref_year)
            return normalize_indicators(
                df, ref, geo_cols,
                minmax_years=case_spec.minmax_years,
                inverse_cols=case_spec.inverse_cols,
            )

        # Minors: remap gender-specific columns to canonical names
        df = df_raw[schema["key_cols"]].copy()
        for canonical in canonical_cols:
            df[canonical] = df_raw[schema["column_roles"][canonical]].values

        # Normalize non-parent geo columns
        parent_cols = {"perc_single_parents_14y", "perc_single_parents_18y"}
        generic_geo = [c for c in geo_cols if c not in parent_cols]
        ref_generic = compute_reference_values(df, df_pop, generic_geo, ref_year=case_spec.ref_year)
        df = normalize_indicators(df, ref_generic, generic_geo, minmax_years=case_spec.minmax_years)

        # Parent columns: shared min/max across gender variants (gender slices)
        # per-column min/max (None slice)
        mask = (df["anno"] >= case_spec.minmax_years[0]) & (df["anno"] <= case_spec.minmax_years[1])

        for suffix in ("14y", "18y"):
            col = f"perc_single_parents_{suffix}"
            if col not in geo_cols:
                continue

            agg_series = df_raw[f"perc_single_parents_{suffix}"]
            ref = compute_reference_values(
                pd.concat([df_raw[schema["key_cols"]], agg_series.rename(col)], axis=1),
                df_pop, [col], ref_year=case_spec.ref_year,
            )[col]

            if slice_spec.gender in {"female", "male"}:
                ref /= 2.0
                other = "male" if slice_spec.gender == "female" else "female"
                shared = pd.concat([
                    df.loc[mask, col],
                    agg_series[mask] / 2.0,
                    df_raw.loc[mask, f"perc_single_{other}_family_{suffix}"],
                ])
                val_min, val_max = shared.min(), shared.max()
            else:
                val_min = df.loc[mask, col].min()
                val_max = df.loc[mask, col].max()

            denom = float(val_max - val_min)
            if denom == 0.0:
                df[col] = 100.0
            else:
                df[col] = 100.0 + 100.0 * (df[col] - ref) / denom

        return df

    def _load_support_data(self, df_indicators: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {
            "indicators": df_indicators,
            "MPI_weights": pd.read_parquet(self.analysis_outputs / "MPI_weights.parquet"),
            "vulnerability_weights": pd.read_parquet(self.analysis_outputs / "vulnerability_weights.parquet"),
            "sensibility_weights": pd.read_parquet(self.analysis_outputs / "sensibility_weights.parquet"),
        }

    @staticmethod
    def load_baseline_mpi(
        *,
        path: Path,
        value_col: str,
        year: int,
        area_id_col: str = "codice_area_statistica",
        year_col: str = "anno",
    ) -> pd.DataFrame:
        df = pd.read_parquet(path)
        df[year_col] = df[year_col].astype(int)
        df = df[df[year_col] == year].copy()

        missing = [c for c in [year_col, area_id_col, value_col] if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required baseline MPI columns: {missing}.")

        df[area_id_col] = df[area_id_col].astype(int)
        return df[[year_col, area_id_col, value_col]].reset_index(drop=True)

    def _compute_accept_zone(
        self,
        df_raw: pd.DataFrame,
        df_pop: pd.DataFrame,
    ) -> dict[int, float]:
        from ..core.accept_zone import compute_accept_zone

        case_spec = self.slice_spec.case_spec
        income_ref_dict = compute_reference_values(
            df_raw, df_pop, ["rmpe_fam"], ref_year=case_spec.ref_year,
        )
        income_ref = income_ref_dict["rmpe_fam"]

        latest = df_raw.groupby("codice_area_statistica")["anno"].max().reset_index()
        latest = latest.merge(
            df_raw[["codice_area_statistica", "anno", "rmpe_fam"]],
            on=["codice_area_statistica", "anno"],
        )
        return compute_accept_zone(
            {int(row.codice_area_statistica): row.rmpe_fam for row in latest.itertuples()},
            income_ref=income_ref,
        )

    @staticmethod
    def _compute_exempt_share(df_raw: pd.DataFrame) -> dict[int, float]:
        latest = df_raw.groupby("codice_area_statistica")["anno"].max().reset_index()
        latest = latest.merge(
            df_raw[["codice_area_statistica", "anno", "perfam_r60"]],
            on=["codice_area_statistica", "anno"],
        )
        return {int(row.codice_area_statistica): float(row.perfam_r60) for row in latest.itertuples()}

    @staticmethod
    def _resolve_base_dir(base_dir: Path | None = None) -> Path:
        if base_dir is not None:
            return Path(base_dir)
        # parents[0]=loading/, ..., [6]=repo root
        return Path(__file__).resolve().parents[6]

    @staticmethod
    def _reshape_to_area_first(shape_a: dict) -> dict:
        shape_b = {}
        for indicator, year_map in shape_a.items():
            for year, area_map in year_map.items():
                for area, index in area_map.items():
                    shape_b.setdefault(area, {}).setdefault(year, {})[indicator] = index
        return shape_b

    @staticmethod
    def _build_index_from_schema(df: pd.DataFrame, schema: dict) -> dict:
        result = {}

        key_cols = schema["key_cols"]
        prefix = schema["name_prefix"]
        nested_key_order = list(schema.get("nested_key_order", key_cols))

        if "value_cols" in schema:
            value_cols = schema.get("canonical_value_cols", schema["value_cols"])
            df = df.melt(
                id_vars=key_cols,
                value_vars=value_cols,
                var_name="indicator",
                value_name="value",
            )
            key_cols = key_cols + ["indicator"]
            value_col = "value"
        else:
            value_col = schema["value_col"]

        if set(nested_key_order) != set(key_cols):
            raise ValueError(
                f"nested_key_order={nested_key_order} must contain exactly the same keys as key_cols={key_cols}"
            )

        for _, row in df.iterrows():
            keys = []

            for col in nested_key_order:
                value = row[col]
                if col.lower() in {"year", "anno"}:
                    value = int(value)
                if col.lower() == "codice_area_statistica":
                    value = int(value)
                keys.append(value)

            cell_value = row[value_col]
            current = result
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            current[keys[-1]] = ConstIndex(
                f"{prefix}[{','.join(map(str, keys))}]",
                cell_value,
            )

        return result
