"""Exercise focused breeding and compact checkpoints through real crash recovery."""
from __future__ import annotations

import json
from pathlib import Path

from phase_17_smoke import main

if __name__ == "__main__":
    config = json.loads(Path("examples/research_experiment003_focused_pilot.json").read_text(
        encoding="utf-8"))
    config.update(name="Phase 17.2 engineering smoke", cycles=2, exact_budget=28,
                  candidate_budget=200, pool_size=8, batch_size=2, checkpoint_every=4,
                  wall_seconds=600,
                  physics={"disorder_widths": [.4], "disorder_seeds": [17001, 17002]},
                  surrogate={"ensemble_size": 4, "n_estimators": 10})
    config["space"].update(side=4, initialization_rewires=[5, 10, 20, 30, 40])
    main(config)
