# evidence/ (local only)

Logs, measurements and images from earlier gates, kept exactly as recorded:

- `G1/` board bring-up, `G2/` Mac sim twin, `G3/` board robotd, `G4/` board camera + YOLO + ball follow
- `walk_drift_20261004/` yaw-drift analysis, `training_20261003/` cloud training run
- `task1_serial_boot/` factory boot logs over the debug serial port

Paths written inside these files are the old ones (`task/evidence/...`, `work/...`); the move log in
`docs/reorg/move_log.tsv` maps them to the new ones. `MANIFEST.tsv` lists every file with SHA-256.
