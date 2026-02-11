Phase 6 semantics repair (opus-agent)
===================================

- [x] Locate and inspect Phase 6 related classes: `AutotuneDemo`, `TuningProfile`, `ProfileStore`, `ProfileValidator`.
- [x] Add `ProfileStore.createFrom(String, ConvergenceReport)` to preserve Phase 5 truth.
- [x] Update `AutotuneDemo` to pass the `ConvergenceReport` into Phase 6 persistence.
- [x] Add unit test `ProfilePersistenceTest` asserting a NOT_CONVERGED report persists with `"converged":false`.
- [x] Built project (compile verified in local Gradle run).

Notes:
- No Phase 5 logic, thresholds, or heuristics were changed.
- `ProfileValidator` was not modified; it does not mutate convergence state.
