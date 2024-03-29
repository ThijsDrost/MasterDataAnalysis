from .General import Analysis, Data_handling, Latex, Plotting, Simulation

__all_exports = [Analysis, Data_handling, Latex, Plotting, Simulation]

for _e in __all_exports:
    _e.__module__ = __name__

__all__ = [e.__name__ for e in __all_exports]
