from src.problab._events import _Event


def _validate_event(value: _Event) -> None:
    if not isinstance(value, _Event):
        raise TypeError("'event' must be an Event.")


def _validate_given(value: _Event | None) -> None:
    if value is not None and not isinstance(value, _Event):
        raise TypeError("'given' must be an Event or None.")
