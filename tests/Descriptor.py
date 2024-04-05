from dataclasses import dataclass
import inspect

from General.Descriptors import Float, Literal, Positive, Integer, Default, SmallerThan, BiggerThan, NoValue, AnyNumber


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
    x: float = Positive() - BiggerThan(10.0)
    y: float = Float() + Literal((1.0, 2.0, 3.0))
    z: int = Integer() + Default(-2) + AnyNumber() - Positive()
    literal: str = Default('c') + Literal(('a', 'b', 'c')) + Literal(('c', 'd', 'e'))


class Testerino:
    value = Integer() + Positive() + SmallerThan(10)

    def __init__(self, value):
        self.value = value

    def printerion(self):
        print(self.value)


data = Data(1, 8., 3.0)
print(data.z)
print(data.literal)


value = Testerino.value
tester = Testerino(5)
tester.printerion()

