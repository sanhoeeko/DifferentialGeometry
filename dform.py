from collections import Counter
from copy import deepcopy

import sympy as sy
from sympy.abc import x, y, z, t
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol


class DBasis:
    def __init__(self, name, index, associated_symbol):
        self.index = index
        self.name = name
        self.symbol = associated_symbol

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.index == other.index

    def __hash__(self):
        return self.index


def inversion(lst: list[int]):
    s = 0
    for i in range(len(lst)):
        for j in range(i):
            if lst[i] < lst[j]:
                s += 1
    return s


def sort_basis(basis: list[DBasis]):
    """
    :return: sort basis by their indices and return the odd(-1), even(1) or duplicated(0) of the permutation
    """
    if len(set(basis)) != len(basis):
        return basis, 0
    sorted_basis = deepcopy(basis)
    sorted_basis.sort(key=lambda x: x.index)
    indices = [item.index for item in basis]
    odd = inversion(indices) % 2
    return sorted_basis, -1 if odd else 1


class DForm(Symbol):
    def __new__(cls, *one_forms: DBasis):
        obj = super().__new__(cls, '∧'.join(list(map(str, one_forms))), commutative=False)
        obj.forms = list(one_forms)
        return obj

    def __mul__(self, other):
        if isinstance(other, DForm):
            forms = self.forms + other.forms
            sorted_forms, coefficient = sort_basis(forms)
            return coefficient * DForm(*sorted_forms)
        return super().__mul__(other)

    def index(self):
        # tuple can be compared large or less by top-down order
        return tuple(map(lambda x: x.index, self.forms))
    
    def _latex(self, printer):
        return r'\wedge '.join(self.name.split('∧'))



dt = DForm(DBasis('dt', 0, t))
dx = DForm(DBasis('dx', 1, x))
dy = DForm(DBasis('dy', 2, y))
dz = DForm(DBasis('dz', 3, z))


class Field(Symbol):
    def __new__(cls, field_name: str, *variables):
        vstrs = list(map(str, variables))
        vstrs.sort()
        obj = super().__new__(cls, f'({field_name})_{{{"".join(vstrs)}}}')
        obj.field_name = field_name
        obj.derivative_variables = Counter(variables)
        return obj

    def __repr__(self):
        return self.name

    def take_derivative(self, variable):
        return Field(self.field_name, *(self.derivative_variables + Counter([variable])))
    
    def _latex(self, printer):
        return self.name


def fields(names: str, *variables):
    names = names.split(' ')
    return [Field(name, *variables) for name in names]


def get_form(mul: Mul):
    form = None
    for a in mul.args:
        if isinstance(a, DForm):
            form = a
            break
    coefficients = list(mul.args)
    coefficients.remove(form)
    coefficient = 1
    for c in coefficients:
        coefficient *= c
    return form, coefficient


def general_derivative(obj, symbol):
    if hasattr(obj, 'take_derivative'):
        return obj.take_derivative(symbol)
    else:
        return sy.diff(obj, symbol)


def _derivative_mul(mul_or_symbol, symbol):
    if isinstance(mul_or_symbol, Mul):
        lst = []
        for i in range(len(mul_or_symbol.args)):
            obj = mul_or_symbol.args[i]
            derive = general_derivative(obj, symbol)
            for j in range(len(mul_or_symbol.args)):
                if j == i: continue
                derive *= mul_or_symbol.args[j]
            lst.append(derive)
        return sum(lst)
    else:
        return general_derivative(mul_or_symbol, symbol)


def ext_d(expr, all_forms: list[DForm]):
    if isinstance(expr, Add):
        return sum([ext_d(a, all_forms) for a in expr.args])
    elif isinstance(expr, DForm):
        return 0
    elif isinstance(expr, Mul):
        dform, coefficient = get_form(expr)
        derivatives = []
        new_forms = []
        all_basis = list(map(lambda x: x.forms, all_forms))
        all_basis = list(set().union(*all_basis))
        for dv in all_basis:
            dx_dv = DForm(dv) * dform
            if dx_dv == 0:
                continue
            new_forms.append(dx_dv)
            derivatives.append(_derivative_mul(coefficient, dv.symbol))
        s = 0
        for i in range(len(derivatives)):
            s += derivatives[i] * new_forms[i]
        return s
    else:
        raise ValueError


def collect(expr: Add):
    expr = sy.simplify(expr)
    forms = []
    coefficients = []
    for a in expr.args:
        form, coefficient = get_form(a)
        if form in forms:
            idx = forms.index(form)
            coefficients[idx].append(coefficient)
        else:
            forms.append(form)
            coefficients.append([coefficient])
    lst = list(zip(forms, coefficients))
    lst.sort(key=lambda x: x[0].index())
    s = 0
    for form, coefficient in lst:
        s += sum(coefficient) * form
    return s


if __name__=='__main__':
    phi, A1, A2, A3 = fields(r'\phi A^{1} A^{2} A^{3}')
    A = phi * dt + A1 * dx + A2 * dy + A3 * dz
    F = ext_d(A, [dx, dy, dz, dt])
    F = collect(F)
    print(F)
