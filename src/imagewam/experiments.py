"""Public experiment names; pose means relative rotation only."""
EXPERIMENTS = {
    'baseline': (False, False),
    'only_pose': (True, False),
    'only_depth': (False, True),
    'both': (True, True),
    'geometry': (True, True),  # Backward-compatible alias for both.
}
