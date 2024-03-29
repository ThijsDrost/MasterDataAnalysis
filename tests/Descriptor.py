from dataclasses import dataclass

from General.Descriptors import Float, Literal, Positive, Integer, Default


@dataclass
class Data:
    """
    Data class with integer, float, and literal attributes

    Parameters
    ----------
    a : int
        An integer attribute
    x : float
        A float attribute
    y : float
        A float attribute
    z : int
        An integer attribute
    literal : str
        A literal attribute
    """
    a: int = Integer()
    x: float = Float() + Positive()
    y: float = Float() + Literal([1.0, 2.0, 3.0])
    z: int = Integer() + Default(4)
    literal: str = Default('c') + Literal(['a', 'b', 'c']) + Literal(['a', 'c'])


data = Data(1, 2.3, 2.0)
data.z
