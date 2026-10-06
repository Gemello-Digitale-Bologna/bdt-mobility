from civic_digital_twins.dt_model import Index


def compute_mpi_index_by_area(
    values_by_area,
    area,
    year,
    mpi_weights,
    *,
    prefix="I_modified_",
    mpi_name="I_modified_MPI",
):
    if area not in values_by_area or year not in values_by_area[area]:
        return None

    data = values_by_area[area][year]
    mpi_value = None

    for indicator_name, weight_index in mpi_weights.items():
        key = f"{prefix}{indicator_name}"
        if key not in data:
            continue

        term = weight_index * data[key]
        mpi_value = term if mpi_value is None else mpi_value + term

    if mpi_value is None:
        return None

    return {
        mpi_name: Index(
            f"{mpi_name}[{area},{year}]",
            mpi_value,
        )
    }
