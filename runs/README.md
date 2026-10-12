# runs/ (local only)

Every launcher writes a new folder `runs/<UTC YYYYMMDDTHHMMSSZ>_<name>/` with its console log and the
evidence copied back from the board. `current_sim_body_log` holds the log path of the latest sim body.
Results worth keeping are summarised into `results/` (tracked) or copied to `evidence/`.
