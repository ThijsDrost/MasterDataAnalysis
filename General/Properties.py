from General.Data_handling import check_literal, check_positive_int, check_positive_float


def get(name: str):
    def _get(self):
        return getattr(self, name)

    return _get


def set_positive_int(name):
    def _set_positive_int(self, value):
        check_positive_int(value, name.removeprefix('_'))
        setattr(self, name, value)

    return _set_positive_int


def set_positive_float(name):
    def _set_positive_float(self, value):
        check_positive_float(value, name.removeprefix('_'))
        setattr(self, name, value)

    return _set_positive_float


def set_literal(literals, name):
    def _set_literal(self, value):
        check_literal(value, literals, name.removeprefix('_'))
        setattr(self, name, value)

    return _set_literal


def property_literal(literals, name, doc=None):
    return property(get(name), set_literal(literals, name), doc=doc)


def property_positive_int(name, doc=None):
    return property(get(name), set_positive_int(name), doc=doc)


def property_float(name, doc=None):
    return property(get(name), set_positive_float(name), doc=doc)
