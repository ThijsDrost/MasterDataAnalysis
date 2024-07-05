from General.simulation.bolsig._bolsig_minus import BolsigMinus

__all_exports = [BolsigMinus]

for _e in __all_exports:
    _e.__module__ = __name__

__all__ = [e.__name__ for e in __all_exports]