# Contributing

Useful contributions include reproducible bug reports, missing-asset reports, independent replications, verified baselines, and carefully scoped alternative interpretations.

Do not modify frozen evidence, old test locks, numerical tolerances, or recorded decisions in place. A change to a model, simulator, loss, optimizer, schedule, budget, or data split creates a new study identity. Record what was changed and preserve the comparison conditions.

For documentation changes, run the standard-library evidence tests. For source refactoring, demonstrate unchanged outputs/gradients against the relevant frozen implementation before claiming equivalence. New training and GPU use require an explicit protocol and resource authorization; CI is not an experiment scheduler.

Do not submit private logs, credentials, copyrighted third-party assets without rights, or executable checkpoint payloads from unknown sources. A result that does not reproduce is welcome when its inputs, environment, limits and failures are recorded.
