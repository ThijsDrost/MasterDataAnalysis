from abc import ABC, abstractmethod
import warnings


class Infinity:
    def __eq__(self, other):
        return isinstance(other, Infinity)

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return isinstance(other, Infinity)

    def __gt__(self, other):
        return True

    def __ge__(self, other):
        return True

    def __format__(self, format_spec):
        return '∞'


class NegativeInfinity:
    def __eq__(self, other):
        return isinstance(other, NegativeInfinity)

    def __lt__(self, other):
        return isinstance(other, Infinity)

    def __le__(self, other):
        return True

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return isinstance(other, NegativeInfinity)

    def __format__(self, format_spec):
        return '-∞'


class NumberLine:
    def init(self, start: float = NegativeInfinity, end: float = Infinity):
        self.ranges = [(start, end)]

    def add(self, start: float = NegativeInfinity, end: float = Infinity):
        new_ranges = []
        for i, (s, e) in enumerate(self.ranges):
            if e < start or end < s:
                pass
            elif start <= s and e <= end:
                new_ranges.append((start, end))
            elif start <= s and end <= e:
                new_ranges.append((s, end))
            elif s <= start and e <= end:
                new_ranges.append((start, e))

    def in_range(self, value: float):
        for s, e in self.ranges:
            if s <= value <= e:
                return True
        return False

    def __contains__(self, item):
        return self.in_range(item)

    def __bool__(self):
        return bool(self.ranges)

class ValidatorDescriptor(ABC):
    _can_second = True

    def __init__(self):
        self._validators = []

    def __set_name__(self, owner, name):
        self.owner = owner
        self.name = name
        self.private_name = f'_{name}'

    def __get__(self, instance, owner):
        return getattr(instance, self.private_name)

    def __set__(self, instance, value):
        self._validate(value)
        setattr(instance, self.private_name, value)

    def __add__(self, other):
        if not other._can_second:
            if not self._can_second:
                raise ValueError(f'Cannot add two ({self.__class__.__name__}, {other.__class__.__name__}) non-second descriptors')
            return other.__add__(self)

        for validator in other._validators:
            for validator2 in self._validators:
                if type(validator) == type(validator2):
                    validator.same(validator2)
                elif any((base in type(validator).__bases__) for base in type(validator2).__bases__):
                    validator.same(validator2)
        self._validators.extend(other._validators)
        return self

    @staticmethod
    def validator(self, value, name=None):
        pass

    def _validate(self, value):
        for validator in self._validators:
            result = validator.validator(validator, value, None)
            if result is not None:
                raise result

    def _validate_message(self, value, name=None):
        for validator in self._validators:
            result = validator.validator(validator, value, name=name)
            if result is not None:
                return result

    def same(self, other):
        pass


class BiggerThan(ValidatorDescriptor):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self._validators.append(self)

    @staticmethod
    def validator(self, value, name=None):
        if value <= self.value:
            if name is None:
                name = self.name
            return ValueError(f'{name} must be bigger than {self.value}')


class Positive(ValidatorDescriptor):
    def __init__(self):
        super().__init__()
        self._validators.append(self)

    @staticmethod
    def validator(self, value, name=None):
        if value <= 0:
            if name is None:
                name = self.name
            return ValueError(f'{name} must be positive')


class IsType(ValidatorDescriptor):
    def __init__(self, type_):
        super().__init__()
        if isinstance(type_, type):
            self.type_ = (type_,)
        elif isinstance(type_, tuple):
            if not all(isinstance(t, type) for t in type_):
                raise ValueError(f'Type requirements must be types or tuple of types, not {type(type_)}')
            self.type_ = type_
        else:
            raise ValueError(f'Type requirements must be types or tuple of types, not {type(type_)}')
        self._validators.append(self)

    @staticmethod
    def validator(self, value, name=None):
        if not isinstance(value, self.type_):
            if name is None:
                name = self.name
            return ValueError(f'{name} must be of type {self.type_.__name__}')

    def same(self, other):
        if set(self.type_).intersection(other.type_) == set():
            types1 = ', '.join(t.__name__ for t in self.type_)
            types2 = ', '.join(t.__name__ for t in other.type_)
            if len(self.type_) == 1:
                types1 += ','
            if len(other.type_) == 1:
                types2 += ','
            raise ValueError(f'Type requirements ({types1}) and ({types2}) are disjoint')


class Float(IsType):
    def __init__(self):
        super().__init__(float)


class Integer(IsType):
    def __init__(self):
        super().__init__(int)


class Literal(ValidatorDescriptor):
    def __init__(self, literals):
        super().__init__()
        self.literals = set(literals)
        self._validators.append(self)

    @staticmethod
    def validator(self, value, name=None):
        if value not in self.literals:
            if name is None:
                name = self.name
            return ValueError(f'{name} must be one of {self.literals}')

    def same(self, other):
        if self.literals.intersection(other.literals) == set():
            raise ValueError('Literals are disjoint')


class Default(ValidatorDescriptor):
    _can_second = False

    def __init__(self, default):
        super().__init__()
        self.default = default

    def __get__(self, instance, owner):
        return getattr(instance, self.private_name, self.default)

    def __add__(self, other):
        ValidatorDescriptor.__add__(self, other)
        result = self._validate_message(self.default, 'default value')
        if result is not None:
            raise ValueError(f'Default value is invalid') from result
        return self

