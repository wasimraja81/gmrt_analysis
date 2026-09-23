# src/instruments/gmrt/

GMRT-specific knowledge, kept separate from the generic algorithms in `src/engine/` (see
`../../../docs/dev/GWB_PIPELINE_REFACTOR_PLAN.md`, standing rule 8). Antenna table
reading, DUD-antenna resolution, row-index building, GSB/GWB channel-range/frequency
mapping, calibrator naming conventions — anything that's true because of this specific
telescope and observation, not true of interferometric calibration in general.

Nothing in `src/engine/` should import from here; data flows the other direction — this
package reads real GMRT data and hands `src/engine/` plain arrays it already knows how to
work with.

Empty until T5b lands.
