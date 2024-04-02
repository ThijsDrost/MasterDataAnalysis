from __future__ import annotations

from typing import Callable, Any
import warnings

from General.Number_line import NumberLine


class NoVal:
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other

    def __sub__(self, other):
        return -other

    def __rsub__(self, other):
        return other

    def __bool__(self):
        return False

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True

    def __repr__(self):
        return 'NoValue'

    def __str__(self):
        return 'NoValue'


NoValue = NoVal()


class Descriptor:
    def __init__(self, default=NoValue, number_line: NumberLine = NoValue, literals: tuple = NoValue,
                 types: tuple = NoValue, converter: Callable[[Any], Any] = NoValue,
                 validators: tuple[Callable[[Any, str], None]] = NoValue):
        self._default = default
        self._number_line = number_line
        self._literals = literals
        self._types = types
        self._converter = converter
        self._validators = validators
        self.name = 'Default'
        self._update()

    def _update(self):
        if self._number_line is not NoValue:
            if not self._number_line:
                raise ValueError(f'Number line is empty')
        if self._literals is not NoValue:
            self._literals = tuple(set(self._literals))
            if not self._literals:
                raise ValueError(f'Literals are empty')
        if self._types is not NoValue:
            self._types = tuple(set(self._types))
            if not self._types:
                raise ValueError(f'Types are empty')

            if self._literals is not NoValue:
                self._literals = tuple((l for l in self._literals if isinstance(l, self._types)))
                if not self._literals:
                    raise ValueError(f'No literals are of the required type')
                self._types = tuple((t for t in self._types if any(isinstance(l, t) for l in self._literals)))

            if self._number_line is not NoValue:
                if (int not in self._types) and (float not in self._types):
                    self._number_line = NoValue
                    warnings.warn('number_line` is not used because `types` does not contain `int` or `float`')

    def __set_name__(self, owner, name):
        self.owner = owner
        self.name = name
        self.private_name = f'_{name}'

    def __get__(self, instance, owner):
        if self._default is not NoValue:
            return getattr(instance, self.private_name, self._default)
        return getattr(instance, self.private_name)

    def __set__(self, instance, value):
        if self._converter is not NoValue:
            value = self._converter(value)
        self._validate(value)
        setattr(instance, self.private_name, value)

    def __add__(self, other: Descriptor) -> Descriptor:
        if not isinstance(other, Descriptor):
            raise TypeError(f'Cannot add {type(other)} to Descriptor')

        def add_dc(a, b, name):
            if a is not NoValue:
                if b is not NoValue:
                    raise ValueError(f'Cannot add two {name}')
                result = a
            else:
                result = b
            return result

        default = add_dc(self._default, other._default, 'default values')
        converter = add_dc(self._converter, other._converter, 'converters')
        validators = add_dc(self._validators, other._validators, 'validators')

        number_line = self._number_line + other._number_line
        literals = self._literals + other._literals
        types = self._types + other._types

        return Descriptor(default=default, number_line=number_line, literals=literals, types=types, converter=converter,
                          validators=validators)

    def __sub__(self, other: Descriptor) -> Descriptor:
        if not isinstance(other, Descriptor):
            raise TypeError(f'Cannot subtract {type(other)} from Descriptor')

        def sub_dc(a, b, name):
            if a == b:
                result = NoValue
            elif b is not NoValue:
                if a is not NoValue:
                    raise ValueError(f'To remove {name}, both descriptors must have the same {name},'
                                     f'not {a} and {b}')
                else:
                    raise ValueError(f'Cannot remove {name} from a descriptor that does not have a {name}')
            else:
                result = a
            return result

        def sub_nlt(a, b, name):
            if a is not NoValue:
                if b is not NoValue:
                    result = a - b
                else:
                    result = a
            else:
                raise ValueError(f'Cannot remove {name} from a descriptor that does not have a {name}')
            return result

        default = sub_dc(self._default, other._default, 'default value')
        converter = sub_dc(self._converter, other._converter, 'converter')
        validators = sub_dc(self._validators, other._validators, 'validators')

        number_line = sub_nlt(self._number_line, other._number_line, 'number line')
        literals = tuple(sub_nlt(set(self._literals), set(other._literals), 'literals'))
        types = tuple(sub_nlt(set(self._types), set(other._types), 'types'))

        for vals, name in ((number_line, 'number lines'), (literals, 'literals'), (types, 'types')):
            if vals is not NoValue:
                if not vals:
                    raise ValueError(f'{name} is empty, cannot remove all values')

        return Descriptor(default=default, number_line=number_line, literals=literals, types=types, converter=converter,
                          validators=validators)

    def _check_type(self, value):
        if self._types is not NoValue:
            for t in self._types:
                if isinstance(value, t):
                    break
            else:
                raise ValueError(f'{self.name} must be of type {self._types}')

    def _check_literal(self, value):
        if self._literals is not NoValue:
            if value not in self._literals:
                raise ValueError(f'{self.name} must be one of {self._literals}')

    def _check_number_line(self, value):
        if self._number_line is not NoValue:
            if value not in self._number_line:
                raise ValueError(f'{self.name} must be in {self._number_line}')

    def _check_validators(self, value):
        if self._validators is not NoValue:
            for validator in self._validators:
                validator(value, self.name)

    def _validate(self, value):
        try:
            self._check_type(value)
            self._check_literal(value)
            self._check_number_line(value)
            self._check_validators(value)
        except Exception as e:
            if self.name != 'Default':
                raise e
            else:
                raise ValueError(f'Default value is invalid') from e


class BiggerThan(Descriptor):
    def __init__(self, value, inclusive=False):
        number_line = NumberLine.bigger_than_float(value, inclusive)
        super().__init__(number_line=number_line)


class SmallerThan(Descriptor):
    def __init__(self, value, inclusive=False):
        number_line = NumberLine.smaller_than_float(value, inclusive)
        super().__init__(number_line=number_line)


class Positive(Descriptor):
    def __init__(self, include_zero=True):
        number_line = NumberLine.bigger_than_float(0, include_zero)
        super().__init__(number_line=number_line)


class Negative(Descriptor):
    def __init__(self, include_zero=True):
        number_line = NumberLine.smaller_than_float(0, include_zero)
        super().__init__(number_line=number_line)


class IsType(Descriptor):
    def __init__(self, types: tuple):
        super().__init__(types=types)


class Float(IsType):
    def __init__(self):
        super().__init__((float, ))


class Integer(IsType):
    def __init__(self):
        super().__init__((int, ))


class Literal(Descriptor):
    def __init__(self, literals):
        super().__init__(literals=literals)


class Default(Descriptor):
    def __init__(self, default):
        super().__init__(default=default)

