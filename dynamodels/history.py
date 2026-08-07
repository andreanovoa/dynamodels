

import numpy as np

__all__ = ['HistoryTracker']


class HistoryTracker:
    """Pre-allocated, growable storage for a state (and time) history.

    Backs `dynamodels.model.Model` and bias estimators in downstream packages:
    states are written into two pre-allocated arrays, `_hist` (shape
    ``(capacity, N, m)``) and `_hist_t` (shape ``(capacity,)``), so that
    appending new states (the common case, one per forecast step) does not
    reallocate on every call. `current_ti` tracks the index of the next empty
    slot; `hist` / `hist_t` expose only the ``[:current_ti]`` valid prefix.
    When the buffer fills up, `_increase_hist_size` grows it in bulk.
    """

    # ________________________ History accessors ________________________ #

    @property
    def hist(self):
        """ndarray: The valid (non-empty) portion of the state buffer,
        `_hist[:current_ti]`, shape ``(current_ti, N, m)``."""
        return self._hist[:self.current_ti]

    @property
    def hist_t(self):
        """ndarray: The valid portion of the time buffer,
        `_hist_t[:current_ti]`, shape ``(current_ti,)``."""
        return self._hist_t[:self.current_ti]

    @property
    def capacity(self):
        """int: Total number of pre-allocated slots, ``_hist_t.shape[0]``
        (>= `current_ti`)."""
        return self._hist_t.shape[0]

    @property
    def current_state(self):
        """ndarray: Most recent state, ``hist[current_ti - 1]``."""
        return self.hist[self.current_ti - 1]

    @property
    def current_time(self):
        """float: Time stamp of `current_state`."""
        return self.hist_t[self.current_ti - 1]

    @property
    def current_ti(self):
        """int: Index of the next empty slot in the buffer (i.e. the number
        of valid entries currently stored)."""
        return self._ti

    @current_ti.setter
    def current_ti(self, value: int):
        self._ti = value


    def __init__(self, initial_capacity=1000):
        """Initialise an empty tracker (the storage arrays themselves are
        allocated lazily, by the first ``update_history(..., reset=True)``
        call).

        Parameters
        ----------
        initial_capacity : int
            Initial capacity of the history arrays.
        """

        self._initial_capacity = initial_capacity
        self.current_ti = 0  # Current time index in history


    def _reset_history(self, new_history, t_reset=None):
        """Resets the history arrays to the provided new_history and t_reset.

        Parameters
        ----------
        new_history : ndarray, shape (Nt, N, m)
            New state history to set.
        t_reset : ndarray, shape (Nt,), optional
            New time history to set. Defaults to zeros, i.e. a reset restarts
            the clock, matching `Model.__init__`.
        """

        if t_reset is None:
            t_reset = np.zeros(new_history.shape[0])

        Nt = max(new_history.shape[0], self._initial_capacity)

        # Initialize the history arrays
        # Preserve dtype: complex-state models (e.g. KS Fourier modes) must not
        # be truncated to their real part.
        self._hist = np.empty((Nt, new_history.shape[1], new_history.shape[2]),
                              dtype=new_history.dtype)
        self._hist_t = np.empty((Nt,))
        # Store the reset history

        self._hist[:new_history.shape[0]] = new_history
        self._hist_t[:t_reset.shape[0]] = t_reset
        self.current_ti = new_history.shape[0]


    def _reset_last_states(self, new_state, t=None):
        """Resets only the last state in the history arrays to the provided new_state and t."""

        # A single state may arrive as (N, m) -- e.g. the analysis ensemble from
        # Estimator.analysis_step. Promote it, otherwise Nt below picks up the state
        # dimension and clobbers that many timesteps.
        if new_state.ndim == 2:
            new_state = new_state[np.newaxis, ...]
        if t is not None:
            t = np.atleast_1d(t)

        Nt = new_state.shape[0]

        # current_ti points at the next empty slot, so the last Nt stored states
        # occupy [current_ti - Nt, current_ti) -- the same window that the `hist`
        # property returns as hist[-Nt:]. Writing one slot earlier leaves the
        # newest state stale and shifts every other state back by one step.
        ti_0 = self.current_ti - Nt
        ti_now = self.current_ti

        assert ti_0 >= 0, f"Cannot reset last {Nt} states because the new time goes to zero. Consider resetting the full history instead."

        self._hist[ti_0:ti_now] = new_state[:]

        if t is not None:
            assert t.shape[0] == Nt, f"Length of t ({t.shape}) must match number of time steps in new_state ({new_state.shape})."
            self._hist_t[ti_0:ti_now] = t[:]




    def update_history(self,
                       state: np.ndarray, t: np.ndarray | None,
                       reset=False, modify_saved_states=False):
        """Write `state` into the history buffer.

        Three mutually exclusive modes, selected by `reset` /
        `modify_saved_states`:

        - `reset=True`: replace the entire buffer with `state` (via
          `_reset_history`), re-allocating storage sized to
          ``max(state.shape[0], initial_capacity)``.
        - `modify_saved_states=True`: overwrite the most recent
          ``state.shape[0]`` already-stored entries in place (via
          `_reset_last_states`), without advancing `current_ti`.
        - otherwise (default): append `state` as new entries starting at
          `current_ti`, growing the buffer first (via `_increase_hist_size`)
          if it would not fit.

        Parameters
        ----------
        state : ndarray
            State(s) to write. In append mode (the default) must have shape
            ``(Nt, N, m)`` with ``N`` matching the existing buffer.
        t : ndarray or None
            Time stamp(s) matching `state`. Required in append mode; optional
            when resetting (defaults to zeros) or modifying saved states.
        reset : bool
            If True, replace the entire history (see above).
        modify_saved_states : bool
            If True, overwrite recent entries in place instead of appending
            (see above); ignored if `reset` is True.
        """
        if t is not None and state.ndim == 3:
            t = np.atleast_1d(t)
            assert state.shape[0] == t.shape[0], f"Length of t ({t.shape}) must match number of time steps in state ({state.shape})."
        if reset: # Reset the full history
            self._reset_history(state, t)

        elif modify_saved_states: # Update only the last state in history
            self._reset_last_states(new_state=state, t=t)
        else:
            assert t is not None, "Time array t must be provided when adding new states to history."
            assert state.ndim == 3, f"State must have shape (Nt, N, m), but got {state.shape}."
            assert state.shape[1] == self._hist.shape[1], f"State N dimension ({state.shape[1]}) must match history N dimension ({self._hist.shape[1]})."

            t0 = self.current_ti
            t1 = t0 + state.shape[0]

            if t1 > self.capacity:
                self._increase_hist_size(Nt=state.shape[0]*10)

            self._hist[t0:t1] = state
            self._hist_t[t0:t1] = t
            self.current_ti = t1


    def _increase_hist_size(self, Nt=None):
        """
        With this I avoid np.concatenate every time I want to add new data to history.
        """

        if Nt is None:
            Nt = self._initial_capacity

        new_capacity = self.capacity + Nt

        # Create new, larger arrays
        new_hist = np.empty((new_capacity, self._hist.shape[1], self._hist.shape[2]),
                            dtype=self._hist.dtype)
        new_hist_t = np.empty((new_capacity,))

        # Copy existing data (expensive operation, but done rarely)
        new_hist[:self.capacity] = self._hist
        new_hist_t[:self.capacity] = self._hist_t

        # Update attributes
        self._hist = new_hist
        self._hist_t = new_hist_t


