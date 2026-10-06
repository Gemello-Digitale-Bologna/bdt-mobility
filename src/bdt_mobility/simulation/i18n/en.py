"""English translations."""

# ===========================================================================
# Dashboard
# ===========================================================================
dashboard_title = "Digital Twin Studio - Bologna Mobility"
scenario_current = "CURRENT SCENARIO"
scenario_example = "Sample policy"
scenario_header = 'Scenario: "{}"'
scenario_header_modified = 'Scenario: modification of "{}"'

# ===========================================================================
# Buttons
# ===========================================================================
btn_save = "Save scenario"
btn_load = "Load scenario"
btn_compare = ">> Compare"
btn_back = "<< Simulation"
btn_download = "Download data"
btn_confirm = "Confirm"

# ===========================================================================
# Parameter groups
# ===========================================================================
params_policy = "General policy parameters"
params_ticket = "Ticket cost"
params_behavior = "Behavioral parameters"
params_tpm = "LPT interventions"
params_tpm_behavior = "Behavioral parameters on LPT"

# ===========================================================================
# Parameter labels
# ===========================================================================
params_start_time = "Policy start time (in hh:mm)"
params_end_time = "Policy end time (in hh:mm)"
params_exempted = "Exempted vehicles (% of total)"
params_ticket_cost = "Ticket cost Euro {} (in \u20ac)"
params_acceptable_cost = "Acceptable cost threshold (in \u20ac, median)"
params_propensity_forward = "Propensity to anticipate the trip (in hours, median)"
params_propensity_postpone = "Propensity to postpone the trip (in hours, median)"
params_median_forward = "Median time of anticipating (in hours)"
params_median_postpone = "Median time of postponing (in hours)"
params_pt_frequency = "Frequency variation of LPT (in %)"
params_pt_capillarity = "Capillarity variation of LPT (in %)"
params_pt_cost = "Cost variation of LPT (in %)"
params_pt_time = "LPT-drive time penalty variation (in %)"
params_behavior_capillarity = "Importance of capillarity of LPT"
params_behavior_frequency = "Importance of frequency of LPT"
params_behavior_cost = "Importance of cost of LPT"
params_behavior_time = "Importance of LPT-drive time penalty"

# ===========================================================================
# Section headers
# ===========================================================================
header_parameters = "Parameters"
header_inflow = "Inflow"
header_traffic = "Traffic"
header_emissions = "Emissions"
header_indicators = "Indicators"

# ===========================================================================
# Visualization settings
# ===========================================================================
viz_settings = "Visualization settings"
viz_type = "Visualization type"
viz_full_area = "Full Urban Area"
viz_zone_selection = "Zone Selection"
viz_map = "Map"
viz_multi_zone = "Multiple zones"
viz_zones = "Zones"
viz_zones_placeholder = "All zones"
viz_zone = "Zone"
viz_time_interval = "Time interval"
viz_time = "Time"

# ===========================================================================
# Comparison view
# ===========================================================================
scenario_a = "Scenario A"
scenario_b = "Scenario B"

# ===========================================================================
# Download dialog
# ===========================================================================
dialog_download_title = "Download data"
dialog_download_progress = "Downloading {}..."
dialog_download_include_ref = "Include reference data"
dialog_download_include_mod = "Include modified data"
dialog_download_include_zones = "Include zone details"

# ===========================================================================
# Save dialog
# ===========================================================================
dialog_save_title = "Save scenario"
dialog_save_name = "Scenario name"
dialog_save_error_empty = "**Error:** Please select a name for saving."
dialog_save_error_reserved = "**Error:** The name is reserved: choose another name!"
dialog_save_error_exists = "**Error:** The name is already used: choose another name!"

# ===========================================================================
# Load dialog
# ===========================================================================
dialog_load_title = "Load scenario"
dialog_load_warning = "**Warning!** Any parameter changes will be lost."
dialog_load_name = "Scenario name"
dialog_load_placeholder = "No scenario available"

# ===========================================================================
# KPI categories
# ===========================================================================
kpi_flow_category = "Flow (vehicles/day entering the ZOI)"
kpi_traffic_category = "Traffic (max vehicles circulating)"
kpi_modal_category = "Modal shift (vehicles/day)"
kpi_emissions_category = "Emissions (Nox gr/day)"
kpi_revenue_category = "Revenue"

# ===========================================================================
# KPI labels
# ===========================================================================
kpi_flow_base = "Reference flow"
kpi_flow_modified = "Modified flow"
kpi_flow_difference = "Flow difference"
kpi_flow_shifted = "Time-shifted flow"
kpi_traffic_base = "Reference traffic"
kpi_traffic_modified = "Modified traffic"
kpi_traffic_difference = "Traffic difference"
kpi_modal_shifted = "Flow shifted to LPT"
kpi_emissions_base = "Reference emissions"
kpi_emissions_modified = "Modified emissions"
kpi_emissions_difference = "Emissions difference"
kpi_revenue_paying = "Paying vehicles (vehicles/day)"
kpi_revenue_avg_cost = "Average cost per vehicle (\u20ac/day)"
kpi_revenue_collected = "Collected fees (\u20ac/day)"

# ===========================================================================
# Ethical visualization
# ===========================================================================

ethical_select_case = "EthicalLens \U0001f50d on Population:"
ethical_select_gender = "Gender Aggregation:"
ethical_case_elderly = "Elderly"
ethical_case_minors = "Families with minors"
ethical_case_income = "Income"
ethical_gender_aggregate = "Aggregated"
ethical_gender_female = "Female"
ethical_gender_male = "Male"
ethical_delta_mpi_header = "\u0394MPI {case} \u2014 {gender} ({year})"
ethical_delta_mpi_label = "\u0394MPI {case}"

# ===========================================================================
# Plot labels (used in view.py)
# ===========================================================================
plot_time = "Time"
plot_flow_label = "Inflow"
plot_flow_unit = "veh/h"
plot_flow_map_label = "Flow"
plot_traffic_label = "Traffic"
plot_traffic_unit = "vehicles"
plot_traffic_map_unit = "max vehicles"
plot_emissions_label = "Emissions"
plot_emissions_unit = "NOx gr/h"
plot_emissions_map_unit = "NOx gr/day"

# ===========================================================================
# Plot legend (used in plot.py)
# ===========================================================================
plot_reference = "Reference"
plot_modified = "Modified (median)"
plot_zone_code = "Zone code"
