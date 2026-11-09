"""Keep a nominal control cadence without replaying missed deadlines."""


def advance_deadline(previous_due, finished_at, period):
    next_due = previous_due + period
    missed = next_due <= finished_at
    if missed:
        next_due = finished_at + period
    return next_due, missed
