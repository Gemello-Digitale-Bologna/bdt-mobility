import numpy as np
import polars as pl


def fold_matrix_24h(
    matrix,
    time_step: int,
    operation: str = "mean"
):
    num_links, num_bins = matrix.shape
    bin_day = 24 * 3600 // time_step
    
    # Init the matrix
    if operation == "min":
        folded_mat = np.full((num_links, bin_day), np.inf, dtype=np.float32)
    else:
        folded_mat = np.zeros((num_links, bin_day), dtype=np.float32)
    
    # Init the count vector (used for the mean)
    presence_count = np.zeros(bin_day, dtype=np.int32)
    
    # Scan the matrix: daily chuncks
    for start_col in range(0, num_bins, bin_day):
        end_col = min(start_col + bin_day, num_bins)
        chunk = matrix[:, start_col:end_col]
        
        # Compute offset and wrap-around (midnight)
        offset = start_col % bin_day
        chunk_len = chunk.shape[1]
        
        # The check handles cases in which 24*3600 % time_step is non null and cases with incomplete matrix (<24h)
        if offset + chunk_len <= bin_day:
            target_slices = [(offset, offset + chunk_len, 0, chunk_len)]
        else:
            split = bin_day - offset
            target_slices = [
                (offset, bin_day, 0, split),           # End day
                (0, chunk_len - split, split, chunk_len) # Start day
            ]

        # Compute the operation            
        for t_start, t_end, c_start, c_end in target_slices:
            if operation == "sum":
                folded_mat[:, t_start:t_end] += chunk[:, c_start:c_end]
            elif operation == "mean":
                folded_mat[:, t_start:t_end] += chunk[:, c_start:c_end]
                presence_count[t_start:t_end] += 1
            elif operation == "max":
                folded_mat[:, t_start:t_end] = np.maximum(folded_mat[:, t_start:t_end], chunk[:, c_start:c_end])
            elif operation == "min":
                folded_mat[:, t_start:t_end] = np.minimum(folded_mat[:, t_start:t_end], chunk[:, c_start:c_end])

    # Last computation for mean and min
    if operation == "mean":
        folded_mat = np.divide(
            folded_mat, presence_count, 
            out=np.zeros_like(folded_mat), 
            where=presence_count > 0
        )
    elif operation == "min":
        # Ripristiniamo a 0 dove non ci sono dati (erano rimasti a inf)
        folded_mat[folded_mat == np.inf] = 0

    return folded_mat


def melt_matrix(
    matrix, 
    mapping,
    time_step: int = 1,
    t_min: int = 0
):
    # Find elements > 0
    rows, cols = np.where(matrix > 0)
    counts = matrix[rows, cols]
    
    # Create output dataframe
    df_final = pl.from_dict({
        "idx": rows.astype(np.int64),
        "t_offset": cols.astype(np.int64),
        "counts": counts.astype(np.int64)
    })

    # Join with mapping
    return (
        df_final
        .join(mapping, on="idx", how="left")
        .with_columns(t = ((pl.col("t_offset") * time_step) + t_min)  )
        .select(["spatial_unit", "t", "counts"])
    )
