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

    def _update(self):
        if self._number_line is not NoValue:
            if not self._number_line:
                raise ValueError(f'Number line is empty')
        if self._literals is not NoValue:
            # To keep the order of the literals, we need to do it this way instead of using a set
            self._literals = tuple((self._literals[i] for i in range(len(self._literals)) if self._literals[i] not in self._literals[:i]))
            if not self._literals:
                raise ValueError(f'Literals are empty')
        if self._types is not NoValue:
            self._types = tuple(set(self._types))
            if not self._types:
                raise ValueError(f'Types are empty')

            if self._literals is not NoValue:
                old_len = len(self._literals)
                self._literals = tuple((l for l in self._literals if isinstance(l, self._types)))
                if not self._literals:
                    raise ValueError(f'No literals are of the required type')
                if len(self._literals) != old_len:
                    warnings.warn('Some literals are not of the required type, they are removed from `literals`')

                old_len = len(self._types)
                self._types = tuple((t for t in self._types if any(isinstance(l, t) for l in self._literals)))
                if old_len != len(self._types):
                    warnings.warn('Some types are not present in `literals`, they are removed from `types`')

            if self._number_line is not NoValue:
                if (int not in self._types) and (float not in self._types):
                    self._number_line = NoValue
                    warnings.warn('number_line` is not used because `types` does not contain `int` or `float`')

    def __set_name__(self, owner, name):
        # Checking is done here, since this is called when all the descriptors are added together
        self.name = f'Default value for `{name}`'  # Set the name to default, so that the error message is more informative
        if self._default is not NoValue:
            self._validate(self._default)
        self._update()

        self.owner = owner
        self.name = name
        self.private_name = f'_{name}'

    def __get__(self, instance, owner):
        if instance is None:
            return self
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

        def sub_n(a, b):
            if a is NoValue:
                if b is NoValue:
                    return NoValue
                else:
                    warnings.warn(f'Trying to remove number line from a descriptor that does not have a number line, assuming'
                                  f'that the number line is {NumberLine()}')
                    a = NumberLine()
            if b is NoValue:
                return a
            return a - b

        def sub_lt(a, b, name):
            if a is not NoValue:
                if b is not NoValue:
                    result = tuple((l for l in a if l not in b))
                else:
                    result = a
            elif b is not NoValue:
                raise ValueError(f'Cannot remove {name} from a descriptor that does not have a {name}')
            else:
                result = NoValue
            return result

        default = sub_dc(self._default, other._default, 'default value')
        converter = sub_dc(self._converter, other._converter, 'converter')
        validators = sub_dc(self._validators, other._validators, 'validators')

        number_line = sub_n(self._number_line, other._number_line)
        literals = sub_lt(self._literals, other._literals, 'literals')
        types = sub_lt(self._types, other._types, 'types')

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
                raise ValueError(f'{self.name} ({value}) must be one of the following types: {self._tuple_str([t.__name__ for t in self._types])}')

    def _check_literal(self, value):
        if self._literals is not NoValue:
            if value not in self._literals:
                raise ValueError(f'{self.name} ({value}) must be one of the following: {self._tuple_str(self._literals)}')

    def _check_number_line(self, value):
        if self._number_line is not NoValue:
            if value not in self._number_line:
                raise ValueError(f'{self.name} ({value}) must be in {self._number_line}')

    def _check_validators(self, value):
        if self._validators is not NoValue:
            for validator in self._validators:
                validator(value, self.name)

    def _validate(self, value):
        if isinstance(value, Descriptor):
            return
        self._check_type(value)
        self._check_literal(value)
        self._check_number_line(value)
        self._check_validators(value)

    @staticmethod
    def _tuple_str(values):
        if len(values) == 1:
            return f'({values[0]},)'
        return f'({", ".join(str(v) for v in values)})'

    def __repr__(self):
        return f'{self.__class__.__name__}(Default={self._default}, NumberLine={self._number_line}, ' \
               f'Literals={self._literals}, Types={self._types}, Converter={self._converter}, ' \
               f'Validators={self._validators})'

class BiggerThan(Descriptor):
    def __init__(self, value, inclusive=False):
        number_line = NumberLine.bigger_than_float(value, inclusive)
        super().__init__(number_line=number_line)


class SmallerThan(Descriptor):
    def __init__(self, value, inclusive=False):
        number_line = NumberLine.smaller_than_float(value, inclusive)
        super().__init__(number_line=number_line)


class AnyNumber(Descriptor):
    def __init__(self):
        number_line = NumberLine()
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

