context = {
  # Hexagons size
  "h3_resolution": 9,
  
  # Parameters for public transport travel times
  "day_of_week" : "monday",
  "start_time_hour" : 8,
  "end_time_hour" : 22,
  "GTFS_DIR": "gtfs_data",
  "max_walk_meters": 1250,
  "walk_speed_ms": 5000 / 3600,
  "max_travel_time_secs": 7200,
  "GTFS_archive": "tper_bologna_gtfs.zip",
  "GTFS_URL": "https://solweb.tper.it/web/tools/open-data/open-data-detail.aspx?filename=gommagtfsbo",
  
  # Parameters for car travel times
  # Δ_traffic: time lost in the traffic compared to free-flow (paper: 22%)
  "delta_traffic" : 0.22,
  "parking_buffer_sec" : 15 * 60,   # Fixed time added for parking + walking (paper: 15 minutes)
  "osrm_base_url" : "http://router.project-osrm.org",
  "batch_size" : 80,
  
  # Parameters for car dependency estimation
  "tau_sec" : 60 * 60,        # τ = 60 minutes in seconds
  "dt_sec"  : 60,             # integration resolution: 1 minute in seconds
  "t_max"   : 3 * 60 * 60,    # Integration time horizon: 3τ = 180 minutes in seconds
  
  # Parameters for population to hexagons assignment
  
  "osm_residential_tags" : {
        "building": [
            "residential", "apartments", "house", "detached",
            "semidetached_house", "terrace", "dormitory", "yes"
        ]
    },
  
  # Parameters for population to hexagons assignment
  "year" : 2022,
  
  # Parameters for scenario simulation
  "target_coverage" : 0.8
}