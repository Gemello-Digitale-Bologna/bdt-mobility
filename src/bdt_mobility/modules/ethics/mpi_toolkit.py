import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def get_reference_df(df, df_population, ref_year, columns_to_exclude, year_col, geo_col, people_col):
    """
    function to get the reference dataframe given the weight of the resident population and the indicators to aggregate
    Calculate weighted mean of indicators for a reference year.
    df = input dataframe with indicators
    df_population = dataframe with population data
    ref_year = reference year for calculation
    columns_to_exclude = list of columns to exclude from calculation
    year_col = name of the column with the year
    geo_col = name of the column with the geographical identifier
    people_col = name of the column with the number of people 
    """
    df_ref = df[df[year_col] == ref_year].copy()
    df_ref = pd.merge(df_ref, df_population, on=geo_col, suffixes=('', '_residenti'))
    total_residents = df_ref[people_col].sum()
    indicator_cols = [c for c in df.columns if c not in columns_to_exclude]
    weighted_means = {}
    for item in indicator_cols:
        weighted_mean = (df_ref[item] * df_ref[people_col]).sum() / total_residents
        # Specific logic: divide per 2 if the indicator regards single parentsS
        # if item in ['perc_single_parents_14y', 'perc_single_parents_18y']:
        #     weighted_mean = weighted_mean / 2
        weighted_means[item] = weighted_mean
    df_reference = pd.DataFrame(list(weighted_means.items()), columns=['indicator', 'ref_value'])
    return df_reference

def normalize_indicators_vincolato_wide(
    df,
    df_ref,
    minmax_years,
    columns_to_exclude,
    year_col,
    geo_col,
    column_for_norm=None,
    ref_value=None  # could be a numeric value or the name of a column to use as reference
):
    """
    Normalize indicators (wide format) using the constrained min-max method
    with respect to a time series and a reference value.

    Parameters
    ----------
    df : pd.DataFrame
        Contains the columns [year, area_code, indicators...]
    df_ref : pd.DataFrame
        Contains the columns ['indicator', 'ref_value'] with the reference values
    minmax_years : tuple
        Time range for calculating the min-max (e.g., (2021, 2023))
    columns_to_exclude : list
        Columns to exclude from normalization (e.g., ['year', 'area_code'])
    year_col : str
        Name of the column with the year (e.g., 'year')
    geo_col : str
        Name of the column with the geographical code (e.g., 'area_code')
    column_for_norm : str o list, opzionale
        Name or list of columns on which to calculate the common min-max range.
        If None, the range for each indicator is calculated separately.
    ref_value : float o str, opzionale
        - If float: use this value as reference for ALL indicators.
        - If str: name of the column from which to extract the reference value (e.g., mean or aggregated column).
        - If None: use the reference values from `df_ref`.

    Returns
    -------
    pd.DataFrame
        DataFrame with the normalized indicators (_norm) and the original ones.
    """

    df_norm = df.copy()
    indicator_cols = [c for c in df.columns if c not in columns_to_exclude]

    # Filter the DataFrame for the specified time range and select relevant columns
    mask_years = (df[year_col] >= minmax_years[0]) & (df[year_col] <= minmax_years[1])
    df_range = df.loc[mask_years, [year_col, geo_col] + indicator_cols]

    # Calculate constrained min and max
    if column_for_norm is not None:
        if isinstance(column_for_norm, list):
            # Min e max global
            val_min = df_range[column_for_norm].min().min()
            val_max = df_range[column_for_norm].max().max()
        else:
            # Single reference column
            val_min = df_range[column_for_norm].min()
            val_max = df_range[column_for_norm].max()
    else:
        # Min e max per each indicator
        val_min = df_range[indicator_cols].min()
        val_max = df_range[indicator_cols].max()

    # Dictionary of reference values
    ref_dict = df_ref.set_index('indicator')['ref_value'].to_dict()

    # Constrained normalization
    for ind in indicator_cols:
        df.loc[:, ind] = pd.to_numeric(df[ind], errors='coerce')

        # Determine the reference value
        if isinstance(ref_value, (int, float)):
            val_ref = ref_value
        elif isinstance(ref_value, str):
            match = df_ref.loc[df_ref['indicator'] == ref_value, 'ref_value']
            val_ref = match.values[0] if not match.empty else np.nan
        else:
            val_ref = ref_dict.get(ind, np.nan)

        # Calculate normalized value
        if np.isscalar(val_min) and np.isscalar(val_max):
            df_norm[ind + '_norm'] = 100 + 100 * ((df[ind] - val_ref) / (val_max - val_min))
        else:
            df_norm[ind + '_norm'] = 100 + 100 * ((df[ind] - val_ref) / (val_max[ind] - val_min[ind]))

    return df_norm

def get_MPI_index_as(
    df, 
    id, 
    year, 
    df_weights=None, 
    df_directions=None
):
    """
    Goal:
        Computes the weighted Multidimensional Poverty Index (MPI) for a specific 
        statistical area and year based on provided weights and semantic directions.

    Input:
        df (pd.DataFrame): Main dataset containing indicators (0–200 range).
        id (str|int): Statistical area code (codice_area_statistica).
        year (int|str): Reference year (anno).
        df_weights (pd.DataFrame): Must contain 'indicator' and 'value' (the weights).
        df_directions (pd.DataFrame, optional): Must contain 'indicator' and 'direction' (direct/inverse).

    Output:
        float|None: The calculated weighted index rounded to 3 decimal places. 
                    Returns None if the area/year combination is not found.
    """

    # 1. Select specific area and year
    # Using .copy() for safety, though we are only reading data
    df_temp = df[(df["codice_area_statistica"] == id) & (df["anno"] == year)].copy()

    if df_temp.empty:
        # Note: this can happen if the area/year does not exist in the integrated dataset
        return None
    
    # 2. Prepare weights (Mandatory)
    if df_weights is None:
        raise ValueError("df_weights is mandatory and must contain 'indicator' and 'value'.")

    # Convert the weights DataFrame into a dictionary {indicator: weight}
    weights_dict = dict(zip(df_weights["indicator"], df_weights["value"]))

    # 3. Prepare semantic directions (Optional)
    directions = {}
    if df_directions is not None:
        directions = dict(zip(df_directions["indicator"], df_directions["direction"]))

    # 4. Calculate weighted index
    mpi_index = 0.0

    for ind, weight in weights_dict.items():
        # Verify that the weighted indicator is actually present in the DataFrame
        if ind not in df_temp.columns:
            # print(f"Warning: indicator '{ind}' missing in integrated dataframe.")
            continue

        # Extract the value (since we filtered by area/year, there is only one value)
        value = df_temp[ind].values[0]

        # Handle direction: if the indicator is "inverse" (e.g., higher is better), 
        # flip it relative to the 200 pivot for fragility scoring.
        if directions.get(ind) == "inverse":
            value = 200 - value 

        mpi_index += weight * value

    return round(mpi_index, 3)

def ampi_vincolato_historical(df, indicator_cols, area_col='codice_area_statistica', year_col='anno',
                              historical_years=None, reference_year=None, penalty_coef=1.0):
    """
    Goal:
        Computes the constrained Adjusted Mazziotta-Versace Index (AMPI) using 
        historical min-max scaling and a penalty factor based on indicator variability.

    Input:
        df (pd.DataFrame): Input data containing area, year, and indicators.
        indicator_cols (list): List of indicator names to be aggregated.
        area_col (str): Column name for the area identifier.
        year_col (str): Column name for the year identifier.
        historical_years (list, optional): Years used to define the historical min-max range.
        reference_year (int, optional): Baseline year to center normalized values.
        penalty_coef (float): Penalty coefficient (multiplies the standard deviation of indicators).

    Output:
        pd.DataFrame: A DataFrame containing the area, year, and the computed AMPI score.
    """
    df = df.copy()
    
    # 1. Filter historical series
    if historical_years is None:
        historical_years = df[year_col].unique()
    
    hist_df = df[df[year_col].isin(historical_years)]
    
    # 2. Calculate historical min and max for each indicator
    mins = hist_df[indicator_cols].min()
    maxs = hist_df[indicator_cols].max()
    
    # 3. Calculate reference mean values for the specified reference year
    if reference_year is None:
        reference_year = historical_years[0]
    ref_mean = df[df[year_col] == reference_year][indicator_cols].mean()
    
    # 4. Define constrained normalization function
    def normalize(row):
        # Normalize values using historical range and shift them based on reference means
        norm_values = ((row[indicator_cols] - mins) / (maxs - mins)) * (ref_mean.max() - ref_mean.min()) + ref_mean.min()
        return norm_values
    
    df[indicator_cols] = df.apply(normalize, axis=1)
    
    # 5. Calculate "horizontal" mean and standard deviation across indicators
    # AMPI formula: Mean + (Penalty Coefficient * Standard Deviation)
    df['ampi'] = df[indicator_cols].mean(axis=1) + penalty_coef * df[indicator_cols].std(axis=1)
    
    return df[[area_col, year_col, 'ampi']]

def categorize_mpi_global_and_gender(df, 
                                              global_col='MPI_eldery_norm',
                                              male_col='MPI_male_eldery_norm',
                                              female_col='MPI_female_eldery_norm',
                                              n_std_levels=[1.0, 0.5],global_mean=None,global_std=None,global_median=None):
    """
    Categorizes elderly MPI for GLOBAL, MALE, and FEMALE using the SAME standard deviation
    calculated from the GLOBAL distribution.

    All three are categorized using the same thresholds, ensuring comparability.

    Parameters:
    -----------
    df : pd.DataFrame
        Wide format with columns: year, statistical_area_code,
                                MPI_elderly_norm, MPI_male_elderly_norm, MPI_female_elderly_norm
    global_col : str
        Name of the normalized global MPI column
    male_col : str
        Name of the normalized male MPI column
    female_col : str
        Name of the normalized female MPI column
    n_std_levels : list
        [outer_std, inner_std] to define categories. Default [1.0, 0.5]

    Returns:
    --------
    df_result : DataFrame with added *_category columns
    stats : dict with global statistics

    Example:
    --------
    df_cat, stats = categorize_elderly_mpi_global_and_gender(
        df,
        global_col='MPI_elderly_norm',
        male_col='MPI_male_elderly_norm',
        female_col='MPI_female_elderly_norm'
    )
    """
    
    df_result = df.copy()
    
    # STEP 1: Compute statistics with the GLOBAL distribution
    # =========================================================
    global_mean = df[global_col].mean() if global_mean is None else global_mean
    global_std = df[global_col].std() if global_std is None else global_std
    global_median = df[global_col].median() if global_median is None else global_median
    
    print(f"\n{'='*70}")
    print(f"GLOBAL REFERENCE STATISTICS (from {global_col})")
    print(f"{'='*70}")
    print(f"Mean:   {global_mean:.2f} (should be ≈100)")
    print(f"Median: {global_median:.2f}")
    print(f"Std:    {global_std:.2f}")
    print(f"\nThese thresholds will be used for GLOBAL, MALE, and FEMALE categorization.")
    
    # STEP 2: Define FIXED bins based on σ global
    # =================================================
    outer_std, inner_std = n_std_levels
    
    bins_fixed = [
        -np.inf,
        100 - outer_std * global_std,    # Very Low threshold
        100 - inner_std * global_std,    # Low threshold
        100 + inner_std * global_std,    # High threshold
        100 + outer_std * global_std,    # Very High threshold
        np.inf
    ]
    
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    
    print(f"\nFixed Category Thresholds (centered on 100):")
    print(f"  Very Low:  < {bins_fixed[1]:.2f}")
    print(f"  Low:       {bins_fixed[1]:.2f} to {bins_fixed[2]:.2f}")
    print(f"  Medium:    {bins_fixed[2]:.2f} to {bins_fixed[3]:.2f}")
    print(f"  High:      {bins_fixed[3]:.2f} to {bins_fixed[4]:.2f}")
    print(f"  Very High: > {bins_fixed[4]:.2f}")
    print(f"{'='*70}\n")
    
    # STEP 3: Apply the SAME thresholds to Global, Male, Female
    # =====================================================
    
    # Categorize GLOBAL
    df_result[f'{global_col}_category'] = pd.cut(
        df[global_col],
        bins=bins_fixed,
        labels=labels,
        include_lowest=True
    )
    
    # Categorize MALE
    df_result[f'{male_col}_category'] = pd.cut(
        df[male_col],
        bins=bins_fixed,
        labels=labels,
        include_lowest=True
    )
    
    # Categorize FEMALE
    df_result[f'{female_col}_category'] = pd.cut(
        df[female_col],
        bins=bins_fixed,
        labels=labels,
        include_lowest=True
    )
    
    # STEP 4: Report the distributions per each
    # ==========================================
    print("Distribution across categories:\n")
    
    for col_name, col in [('GLOBAL', global_col), ('MALE', male_col), ('FEMALE', female_col)]:
        print(f"{col_name} ({col}):")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Std:  {df[col].std():.2f}")
        
        dist = df_result[f'{col}_category'].value_counts().sort_index()
        for cat in labels:
            count = dist.get(cat, 0)
            pct = (count / len(df) * 100) if len(df) > 0 else 0
            print(f"  {cat:12s}: {count:4d} ({pct:5.1f}%)")
        print()
    
    # STEP 5: Gender comparison
    # =========================
    print(f"{'='*70}")
    print("GENDER COMPARISON")
    print(f"{'='*70}")
    
    male_mean = df[male_col].mean()
    female_mean = df[female_col].mean()
    gap = female_mean - male_mean
    
    print(f"Mean MVI:")
    print(f"  Male:   {male_mean:.2f}")
    print(f"  Female: {female_mean:.2f}")
    print(f"  Gap:    {gap:+.2f} ({'Female higher' if gap > 0 else 'Male higher'})")
    
    # Chi-square test
    from scipy.stats import chi2_contingency
    
    # Create contingency table per gender
    # First, convert in long format for the test
    df_male_long = df_result[['codice_area_statistica', f'{male_col}_category']].copy()
    df_male_long['gender'] = 'male'
    df_male_long.rename(columns={f'{male_col}_category': 'category'}, inplace=True)
    
    df_female_long = df_result[['codice_area_statistica', f'{female_col}_category']].copy()
    df_female_long['gender'] = 'female'
    df_female_long.rename(columns={f'{female_col}_category': 'category'}, inplace=True)
    
    df_gender = pd.concat([df_male_long, df_female_long], ignore_index=True)
    
    contingency = pd.crosstab(df_gender['gender'], df_gender['category'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    print(f"\nChi-square test:")
    print(f"  χ² = {chi2:.2f}, p = {p_value:.4f}")
    print(f"  → {'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'} difference (α=0.05)")
    print(f"{'='*70}\n")
    
    # STEP 6: Store statistics
    # =========================
    stats = {
        'global_mean': global_mean,
        'global_median': global_median,
        'global_std': global_std,
        'bins': bins_fixed,
        'labels': labels,
        'male_mean': male_mean,
        'female_mean': female_mean,
        'gender_gap': gap,
        'chi2': chi2,
        'p_value': p_value
    }
    
    return df_result, stats


def visualize_categories(df_cat, 
                                 global_col='MPI_eldery_norm',
                                 male_col='MPI_male_eldery_norm',
                                 female_col='MPI_female_eldery_norm',
                                 title_suffix=''):
    """
    Compact visualization of Global, Male, Female MPI distributions.
    Optimized spacing and layout to eliminate white space.
    """

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#8e44ad']

    # === GLOBAL ===
    global_dist = df_cat[f'{global_col}_category'].value_counts().reindex(labels, fill_value=0)
    axes[0, 0].bar(labels, global_dist.values, color=colors, alpha=0.8, edgecolor='black')
    axes[0, 0].set_title(title_suffix + 'Global MPI', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Areas', fontsize=10)
    axes[0, 0].tick_params(axis='x', rotation=30, labelsize=9)
    axes[0, 0].grid(axis='y', alpha=0.25)

    for i, v in enumerate(global_dist.values):
        pct = (v / len(df_cat) * 100) if len(df_cat) > 0 else 0
        axes[0, 0].text(i, v + 0.2, f'{v}\n({pct:.1f}%)', 
                        ha='center', va='bottom', fontsize=9)

    # === MALE vs FEMALE DISTRIBUTION ===
    male_dist = df_cat[f'{male_col}_category'].value_counts().reindex(labels, fill_value=0)
    female_dist = df_cat[f'{female_col}_category'].value_counts().reindex(labels, fill_value=0)
    
    x = np.arange(len(labels))
    width = 0.35

    axes[0, 1].bar(x - width/2, male_dist.values, width, label='Male', 
                   color='#3498db', alpha=0.8, edgecolor='black')
    axes[0, 1].bar(x + width/2, female_dist.values, width, label='Female',
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    
    axes[0, 1].set_title('Male vs Female Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=30, fontsize=9)
    axes[0, 1].set_ylabel('Areas', fontsize=10)
    axes[0, 1].grid(axis='y', alpha=0.25)
    axes[0, 1].legend(fontsize=9)

    # === VIOLIN PLOT: MVI DISTRIBUTION (MALE vs FEMALE) ===
    df_violin = pd.DataFrame({
        'category': list(df_cat[f'{male_col}_category']) + list(df_cat[f'{female_col}_category']),
        'MVI': list(df_cat[male_col]) + list(df_cat[female_col]),
        'gender': ['male'] * len(df_cat) + ['female'] * len(df_cat)
    })

    sns.violinplot(
        data=df_violin,
        x='category',
        y='MVI',
        hue='gender',
        split=True,
        ax=axes[1, 0],
        palette=['#3498db', '#e74c3c']
    )

    axes[1, 0].axhline(
        y=df_violin['MVI'].mean(),
        color='black',
        linestyle='--',
        linewidth=1,
        alpha=0.4
    )

    axes[1, 0].set_title('MVI Distribution by Category (Violin)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Severity Category')
    axes[1, 0].set_ylabel('MVI')
    axes[1, 0].tick_params(axis='x', rotation=30, labelsize=9)
    axes[1, 0].grid(axis='y', alpha=0.25)
    axes[1, 0].legend(title='Gender', fontsize=9)

    # === GENDER GAP ===
    df_male_cat = df_cat.groupby(f'{male_col}_category')[male_col].mean().reindex(labels)
    df_female_cat = df_cat.groupby(f'{female_col}_category')[female_col].mean().reindex(labels)
    
    gap_by_cat = df_female_cat - df_male_cat
    colors_gap = ['green' if x < 0 else 'red' for x in gap_by_cat.values]

    axes[1, 1].bar(labels, gap_by_cat.values,
                   color=colors_gap, alpha=0.8, edgecolor='black')
    axes[1, 1].axhline(0, color='black', linewidth=1)
    
    axes[1, 1].set_title('Gender Gap (F - M)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Δ MPI', fontsize=10)
    axes[1, 1].tick_params(axis='x', rotation=30, labelsize=9)
    axes[1, 1].grid(axis='y', alpha=0.25)

    for i, v in enumerate(gap_by_cat.values):
        if not np.isnan(v):
            axes[1, 1].text(i, v + (0.1 if v > 0 else -0.1),
                            f'{v:.2f}', ha='center',
                            va='bottom' if v > 0 else 'top', fontsize=9)

    # === COMPACT LAYOUT ===
    plt.tight_layout(pad=1.0)
    fig.subplots_adjust(
        top=0.92, 
        bottom=0.07,
        left=0.05,
        right=0.98,
        hspace=0.35,
        wspace=0.25
    )

    plt.close(fig)

    return fig
