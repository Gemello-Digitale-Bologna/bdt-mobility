# Glossary

Terms specific to this project — domain concepts, platform conventions, and
codebase vocabulary. Alphabetical order.

---

**BDT / Bologna Digital Twin / Civic Digital Twin**
The umbrella research project in which this package is developed. BDT infrastructure
includes DigitalHub (platform) and the `bdt_mobility` Python package.

**DATA_TAG** (`BDT_DATA_TAG` env var)
A string identifier that selects which version of the dataset to use. The default is
`san_felice` (a Bologna neighbourhood used as the primary test area). Data lives under
`data/<DATA_TAG>/`. Override with `export BDT_DATA_TAG=<tag>` before running any step.

**DigitalHub**
The FBK-operated MLOps/data platform used to orchestrate and store pipeline runs. The
`digitalhub` SDK is used to register functions, trigger workflows, and log artifacts and
dataitems. DigitalHub is the remote execution target; local execution (`run_extraction.py`)
does not require a DigitalHub connection.

**extraction pipeline**
The ordered sequence of `step_*.py` modules that transform raw inputs into analysis
artifacts. Assembled in `extraction/pipeline.py` as `PIPELINE`. Runs in full via
`run_extraction.py` or on the platform via `ingest_data.py`.

**LTZ (Limited Traffic Zone)**
Bologna's existing limited traffic zone (ZTL) in the city centre (`ltz_boundary_{YEAR}`
dataitem, `PathRegistry.LTZ_BOUNDARY`). Distinct from the ZOI.

**ZOI (Zone of Interest)**
The area subject to the access-restriction policy being modeled, defined by a manually
drawn perimeter (`zoi_manual_border` dataitem, `PathRegistry.ZOI_BORDER`). OD zones are
classified as inside or outside the ZOI.
