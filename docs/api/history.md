# `HistoryTracker`

Pre-allocated, growable storage for a state (and time) history. Backs
`Model.hist` / `Model.hist_t`, so that repeated forecast steps do not
reallocate on every call.

::: dynamodels.history.HistoryTracker
