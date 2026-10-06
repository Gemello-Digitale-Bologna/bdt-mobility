"""
Utilities for loading, validating, and merging gate traffic parquet/csv files.

Each file is expected to contain hourly traffic counts per gate and vehicle.
The merging pipeline checks that overlapping time regions across files are
consistent before combining them into a single DataFrame.
"""

import pandas as pd
import plotly.express as px
import numpy as np
from pathlib import Path


from pathlib import Path
import os
from bdt_mobility.utils.config import DATA_PATH

GATE_PATH = DATA_PATH / 'preparation' / 'gate'
EXPORT_PARQUET_FOLDER = GATE_PATH / 'gate_parquet'
# this folder should already exist and contain the parquet data, therefore we don't call mkdir



def merge_files_and_save(
    files: list[str | Path] | None = None,
    folder: str | Path | None = None,
    save_path: str | Path | None = None,
) -> None:
    """
    Merges parquet/csv files and saves the result to `save_path`.

    Convenience wrapper around `merge_files`. See its documentation for
    details on how `files` and `folder` interact.

    The output format is inferred from the `save_path` extension:
    ``.parquet`` files are saved with ``DataFrame.to_parquet``;
    ``.csv`` files are saved with ``DataFrame.to_csv`` (no index).

    Parameters
    ----------
    files
        File names or paths. Treated as relative to `folder` if both are given.
    folder
        Directory to scan or use as base path for `files`.
     save_path
        Destination path for the merged file. Must end with ``.parquet`` or
        ``.csv``; any other extension raises a ``ValueError``.
    """

    if save_path is None:
        raise ValueError('Plese pass the `save_path` parameter')
    suffix = Path(save_path).suffix.lower()
    if suffix not in {".parquet", ".csv"}:
        raise ValueError(f"Unsupported file extension {suffix!r}. `save_path` must end  with '.parquet' or '.csv'.")
    
    df = merge_files(files=files, folder=folder)
    
    if suffix == ".parquet":
        df.to_parquet(save_path)
    else:
        df.to_csv(save_path, index=False)


def merge_files(
    files: list[str | Path] | None = None,
    folder: str | Path | None = None,
) -> pd.DataFrame:
    """
    Loads, validates, and merges files into a single DataFrame.

    Input resolution priority:
    - both `files` and `folder`: looks for each filename relative to the folder
    - only `folder`: loads all files found in the folder
    - only `files`: treats each entry as a full file path

    Parameters
    ----------
    files
        List of file names or paths. If `folder` is also given, these are
        treated as relative paths inside that folder.
    folder
        Directory to scan or use as base path for `files`.

    Notes
    -----
    Raises ValueError if neither argument is provided
    or if files fail the overlap/continuity check.
    """
    
    if files is None and folder is None:
        raise ValueError('al least one of \'files\' or \'folder\' must be defined')
    
    dataframes_raw = load_gates(files=files, folder=folder)
    dataframes = transform_vehicle_independent_dataframes(dataframes_raw)

    if not check_file_overlap_continuity(dataframes.copy()):
        raise ValueError('incompatible files')
    
    merged_df = pd.concat(list(dataframes_raw.values())).groupby(['Data','gate','vehicle']).max().reset_index().sort_values('Data')
    return merged_df


def check_file_overlap_continuity(
    dataframes: dict[str, pd.DataFrame],
    return_detailed_compatibility: bool = False,
    return_overall_compatibility: bool = False,
    plot: bool = False,
) -> bool | pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Checks pairwise overlap compatibility across all loaded gate files.

    For each pair of files, compares common gates using
    `check_overlap_continuity` and aggregates results into compatibility
    matrices.

    Parameters
    ----------
    dataframes
        Dictionary mapping file names to vehicle-independent gate DataFrames.
    return_detailed_compatibility
        If True, returns a tuple of (overall, body, link) compatibility matrices.
    return_overall_compatibility
        If True, returns only the overall compatibility matrix.
    plot
        If True, plots the overall, body, and link compatibility matrices.

    Notes
    -----
    Default return is a single bool: True if all gate pairs across all file
    combinations are compatible.

    Returns overall compatibility matrix, 
    if `return_overall_compatibility` is True.
    
    Returns (overall, body, link) compatibility matrices, if 
    `return_detailed_compatibility` is True.
    """
    if return_detailed_compatibility and return_overall_compatibility:
        raise ValueError('Please select only one of \'return_detailed_compatibility\' or \'return_overall_compatibility\'')
    
    LINK_LENGTH = 4
    file_names = list(dataframes.keys())
    body_compatibility = pd.DataFrame(True, index=file_names, columns=file_names)
    link_compatibility = pd.DataFrame(True, index=file_names, columns=file_names)
    all_gates_compatibility = []

    # computing data compatibility between all files
    for i in range(len(file_names)):
        file1 = file_names[i]
        df1 = dataframes[file1]

        for j in range(i+1, len(file_names)):
            file2 = file_names[j]
            df2 = dataframes[file2]

            common_columns = df1.columns.intersection(df2.columns)

            for gate_name in common_columns:
                s1 = df1[gate_name]
                s2 = df2[gate_name]

                res = check_overlap_continuity(s1, s2, LINK_LENGTH, freq='h')

                if not res['body_compatible']:
                    body_compatibility.loc[file1, file2] = False
                    body_compatibility.loc[file2, file1] = False
                if not res['link_compatible']:
                    link_compatibility.loc[file1, file2] = False
                    link_compatibility.loc[file2, file1] = False
                    
                all_gates_compatibility.append(all(res.values()))
    
    overall_compatibility = body_compatibility & link_compatibility
    all_files_compatibility = all(all_gates_compatibility)

    if plot:
        _plot_compatibility(overall_compatibility, 'overall')
        _plot_compatibility(body_compatibility, 'body')
        _plot_compatibility(link_compatibility, 'link')

    if return_overall_compatibility:
        return  overall_compatibility
    
    if return_detailed_compatibility:
        return overall_compatibility, body_compatibility, link_compatibility
    
    return all_files_compatibility


def check_overlap_continuity(
    s1: pd.Series,
    s2: pd.Series,
    link_length: int,
    freq: str | None = None,
) -> dict[str, bool]:
    """
    Checks whether two overlapping time series are compatible for merging.

    The overlap region is split into two parts:
    - **body**: shared timestamps where both series must be equal
    - **link**: the last `link_length` timestamps of the overlap (transition
      zone), where the earlier-ending series must be <= the later one

    NaN values are ignored in both checks.

    Parameters
    ----------
    s1
        First time series with a DatetimeIndex.
    s2
        Second time series with a DatetimeIndex.
    link_length
        Number of timestamps at the end of the overlap to treat as the
        transition zone.
    freq
        Time frequency of the index. Inferred automatically if not provided.

    Returns
    -------
    Dictionary with keys:
    - `link_compatible`: True if the earlier series is <= the later in the link zone
    - `body_compatible`: True if both series are equal in the overlap body
    """
    if freq is None:
        freq = pd.infer_freq(s1.index)
        if freq is None:
            raise ValueError(
                "Could not automatically infer frequency from index. "
                "Please pass the `freq` parameter explicitly (e.g. freq='D' or freq='h')."
            )
        
    # finding the first and last non NaN value of the series
    start_index1, end_index1 = _first_and_last_value_indices(s1.values)
    start_index2, end_index2 = _first_and_last_value_indices(s2.values)

    end1 = s1.index[end_index1]
    end2 = s2.index[end_index2]

    # finding the series that ends first
    if end2 < end1:
        s1, s2 = s2, s1
        start_index1, start_index2 = start_index2, start_index1
        end_index1, end_index2 = end_index2, end_index1
        end1, end2 = end2, end1

    end_overlap = end1

    start1 = s1.index[start_index1]
    start2 = s2.index[start_index2]

    if start1 > start2:
        start_overlap = start1
    else:
        start_overlap = start2

    # link compatibility
    link_timestamps = pd.date_range(end=end_overlap, freq=freq, periods=link_length)

    s1_link = s1.reindex(link_timestamps)
    s2_link = s2.reindex(link_timestamps)

    link_compatibility = ((s1_link <= s2_link) | s1_link.isna() | s2_link.isna()).all()

    # body compatibility
    body_timestamps = pd.date_range(start=start_overlap, end=end_overlap, freq=freq).difference(link_timestamps)
    
    s1_body = s1.reindex(body_timestamps)
    s2_body = s2.reindex(body_timestamps)

    body_compatibility = ((s1_body == s2_body) | s1_body.isna() | s2_body.isna()).all()
    
    return {
        "link_compatible": link_compatibility,
        "body_compatible": body_compatibility
        }


def load_gates(
    files: list[str | Path] | None = None,
    folder: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Loads files form a folder into a dictionary of DataFrames.

    Parameters
    ----------
    files
        File names or paths. Treated as relative to `folder` if both are given.
    folder
        Directory to scan or use as base path for `files`.
    """

    if folder is not None:
        folder = Path(folder)
    if files is not None:
        files = [Path(f) for f in files]

    if files is not None and folder is not None:
        files = [folder/file for file in files]
    elif folder is not None:
        files = [file for file in folder.iterdir() if file.is_file()]
    elif files is not None:
        files = files

    dataframes_raw = {}

    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix == ".csv":
            df = pd.read_csv(path)
        else:
            continue
        
        df['Data'] = pd.to_datetime(df['Data'])
        dataframes_raw[path.name] = df

    return dataframes_raw


def load_gates_vehicle_independent(
    files: list[str | Path] | None = None,
    folder: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Loads parquet/csv files and aggregates counts across vehicles.

    Convenience wrapper around `load_gates` and
    `transform_vehicle_independent_dataframes`.

    Parameters
    ----------
    files
        File names or paths. Treated as relative to `folder` if both are given.
    folder
        Directory to scan or use as base path for `files`.
    """
    dataframes = load_gates(files, folder)
    dataframes = transform_vehicle_independent_dataframes(dataframes)    
    return dataframes


def transform_vehicle_independent_dataframes(
    dataframes_raw: dict[str, pd.DataFrame],
    freq: str = 'h',
) -> dict[str, pd.DataFrame]:
    """
    Applies `transform_vehicle_independent_df` to each DataFrame in the dict.

    Parameters
    ----------
    dataframes_raw
        Dictionary mapping file names to raw gate DataFrames.
    freq
        Passed through to `transform_vehicle_independent_df`. Default is 'h'.
    """
    dataframes = {}
    for file, df in dataframes_raw.items():
        dataframes[file] = transform_vehicle_independent_df(df, freq)
    return dataframes


def transform_vehicle_independent_df(
    df: pd.DataFrame,
    freq: str = 'h',
) -> pd.DataFrame:
    """
    Aggregates counts by gate and time, summing across vehicles.

    Parameters
    ----------
    df
        Raw gate DataFrame with columns: gate, Data, count.
    freq
        Time frequency passed to asfreq. Default is 'h' (hourly).
    """
    return df.groupby(['gate', 'Data'])['count'].sum().unstack('gate').sort_index().asfreq(freq)


def _first_and_last_value_indices(arr: np.ndarray) -> tuple[int, int]:
    """Returns the indices of the first and last non-NaN values in an array."""
    arr = np.isnan(arr)
    first_valid_value = np.where(arr==0)[0][0]
    last_valid_value = np.where(arr==0)[0][-1]
    return first_valid_value, last_valid_value


def _plot_compatibility(
    compatibility: pd.DataFrame,
    title: str | None = None,
) -> None:
    """Plots a compatibility matrix as a heatmap using plotly."""
    color_scale = ['rgba(255,15,15,0.9)', 'rgba(15,15,255,0.9)']
    fig = px.imshow(compatibility, color_continuous_scale=color_scale, range_color=[0,1])
    fig.update_xaxes(showticklabels=False)
    fig.update_layout({'title':f'{title} compatibility'})
    fig.update_traces(xgap=2, ygap=2)
    fig.show()


def main():
    folder = EXPORT_PARQUET_FOLDER
    save_file = GATE_PATH / 'merged_gates.parquet'
    merge_files_and_save(folder=folder, save_path=save_file)


if __name__ == "__main__":
    main()

# uv run python -m bdt_mobility.modules.transport.preparation.gates.step_02_merge_gate_files