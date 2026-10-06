from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from civic_digital_twins.dt_model import Index

from .cases import GenderKey
from .model import EthicalModel
from .projection import ProjectionMap


@dataclass(frozen=True)
class EthicalBundleOutputs:
    baseline_mpi: dict[GenderKey, dict[int, Index]]
    modified_mpi: dict[GenderKey, dict[int, Index]]
    delta_mpi: dict[GenderKey, dict[int, Index]]


@dataclass(frozen=True)
class EthicalBundleExpose:
    vulnerability_by_area: dict[GenderKey, dict[int, Index]]


class EthicalModelBundle:
    def __init__(
        self,
        *,
        ztl_start: Index,
        ztl_end: Index,
        avail_postpone: Index | None = None,
        avail_anticipate: Index | None = None,
        median_postpone: Index | None = None,
        median_anticipate: Index | None = None,
        ticket_costs: Index | None = None,
        euro_class_split: Index | None = None,
        case_id: str = "elderly",
        genders: tuple[GenderKey, ...] = ("None", "male", "female"),
        year: int = 2023,
        base_dir: Path | None = None,
        projection: ProjectionMap | None = None,
        k_scale: float = 1.0,
    ) -> None:

        # check
        if not genders:
            raise ValueError("EthicalModelBundle requires at least one gender.")

        # instantiate the gender-child models
        _ethical_inputs = EthicalModel.Inputs(
            ztl_start=ztl_start,
            ztl_end=ztl_end,
            avail_postpone=avail_postpone,
            avail_anticipate=avail_anticipate,
            median_postpone=median_postpone,
            median_anticipate=median_anticipate,
            ticket_costs=ticket_costs,
            euro_class_split=euro_class_split,
        )
        self.models = {
            gender: EthicalModel(
                inputs=_ethical_inputs,
                config=EthicalModel.Config(
                    case_id=case_id,
                    gender=gender,
                    year=year,
                    base_dir=base_dir,
                    projection=projection,
                    k_scale=k_scale,
                )
            )
            for gender in genders
        }

        # chose an arbitrary child model to get the "case_spec" form it
        reference_gender = genders[0]
        reference_model = self.models[reference_gender]

        # get the case specs
        self.case_spec = reference_model.case_spec
        self.slice_specs = {gender: model.slice_spec for gender, model in self.models.items()}
        self.output_geography = reference_model.output_geography

        # bundle the outputs of all the child models
        self.outputs = EthicalBundleOutputs(
            baseline_mpi={gender: model.outputs.baseline_mpi for gender, model in self.models.items()},
            modified_mpi={gender: model.outputs.modified_mpi for gender, model in self.models.items()},
            delta_mpi={gender: model.outputs.delta_mpi for gender, model in self.models.items()},
        )
        self.expose = EthicalBundleExpose(
            vulnerability_by_area={gender: model.expose.vulnerability_by_area for gender, model in self.models.items()},
        )

        # create aliases
        self.I_baseline_mpi = self.outputs.baseline_mpi
        self.I_modified_mpi = self.outputs.modified_mpi
        self.I_delta_mpi = self.outputs.delta_mpi
        self.I_vulnerability_by_area = self.expose.vulnerability_by_area
        # bundle indexes for MobilityModel
        self.indexes = [
            index
            for child_model in self.models.values()
            for index in child_model.indexes
        ]
        
    # package all the outputs maps into a one dict keyed also by gender
    def extract_output_maps(self, subs: dict) -> dict[str, dict[GenderKey, dict[int, float]]]:
        child_maps = {
            gender: model.extract_output_maps(subs)
            for gender, model in self.models.items()
        }
        return {
            "baseline_mpi": {
                gender: maps["baseline_mpi"]
                for gender, maps in child_maps.items()
            },
            "modified_mpi": {
                gender: maps["modified_mpi"]
                for gender, maps in child_maps.items()
            },
            "delta_mpi": {
                gender: maps["delta_mpi"]
                for gender, maps in child_maps.items()
            },
        }
