from ._names import Names

__all_exports = [Names]

for _e in __all_exports:
    _e.__module__ = __name__

__all__ = [e.__name__ for e in __all_exports]
