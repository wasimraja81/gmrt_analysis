# src/data_io/

Read-only raw-data access layer (`raw_data_access.py`). Every read of the raw GWB FITS
data goes through this module, opened in a read-only mode enforced centrally rather than
per-callsite. Also owns the output-path safety guard: any function that writes a derived
product must be checked here against the raw input path (hard error on equality), as
defense in depth for the never-modify-raw-data requirement.

Named `data_io`, not `io`, to avoid shadowing Python's own standard-library `io` module.
