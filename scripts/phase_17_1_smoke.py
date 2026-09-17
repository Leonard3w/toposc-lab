"""Run the established hard-crash/recovery smoke with expanded connectivity."""
from __future__ import annotations

import json
from pathlib import Path

from phase_17_smoke import main

if __name__ == "__main__":
    config = json.loads(Path("examples/research_experiment002_long_connectivity.json").read_text(
        encoding="utf-8"))
    config.update(name="Phase 17.1 expanded-connectivity engineering smoke", cycles=2,
                  exact_budget=28, candidate_budget=200, pool_size=8, batch_size=2,
                  checkpoint_every=4, wall_seconds=600,
                  physics={"disorder_widths": [.4], "disorder_seeds": [17001, 17002]},
                  surrogate={"ensemble_size": 4, "n_estimators": 10})
    config["space"]["side"] = 4
    main(config)
