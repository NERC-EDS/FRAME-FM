# data/

Lightweight data used by test scripts and small examples.

This directory is intended for:
- small test fixtures (tiny CSVs, small arrays, minimal raster samples)
- deterministic inputs for unit/integration tests
- lightweight demo data for quick sanity-check runs

This directory is **not** intended for:
- full training datasets
- large cached artefacts
- model checkpoints
- experiment outputs

## Guidelines
- Keep files small (ideally KB–few MB max)
- Prefer open formats (CSV/JSON/NPY/etc.)
- Keep fixtures stable to avoid breaking tests
- Document the purpose of any non-obvious file
