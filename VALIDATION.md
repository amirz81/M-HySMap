# Validation

A clean smoke run was executed from the supplied `mhysmap_v13` snapshot:

```bash
python run_heavy_v12.py   --config configs/smoke_v12.json   --out-dir <temporary-output>   --timeout-per-run 300
```

Observed result:

- jobs: 1
- completed: 1
- return code: 0
- wall time: approximately 5.4 s in the validation environment

This repository does not contain a standalone pytest suite; the experiment harness smoke
configuration is therefore the primary executable sanity check.

A basic text scan found no API keys/secrets or user-specific absolute machine paths.
