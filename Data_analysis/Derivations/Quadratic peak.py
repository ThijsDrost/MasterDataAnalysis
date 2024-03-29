import sympy
from sympy.solvers import solve

x_1, x_2, x_3 = sympy.symbols(r'\lambda_1 \lambda_2 \lambda_3')
y_1, y_2, y_3 = sympy.symbols('I_1 I_2 I_3')
# x_1, x_2, x_3 = sympy.symbols(r'x_1 x_2 x_3')
# y_1, y_2, y_3 = sympy.symbols('y_1 y_2 y_3')
a, b, c = sympy.symbols('a b c')

# %% Center location
result = solve([a*x_1**2 + b*x_1 + c - y_1, a*x_2**2 + b*x_2 + c - y_2, a*x_3**2 + b*x_3 + c - y_3], a, b, c)
print(result[a])
print(result[b])
print(result[c])

peak = -result[b] / (2*result[a])
peak = peak.collect((x_1, x_2, x_3))
print(peak)
print(sympy.latex(peak))
print('a=', sympy.latex(result[a].collect((x_1, x_2, x_3))))
print('b=', sympy.latex(result[b].collect((x_1, x_2, x_3))))

# %% Peak width
e = sympy.symbols('e')
equation = a*x_1**2 + b*x_1 + c
height = equation.subs(x_1, -b/(2*a))
half_height = height * e
half_loc = solve(equation - half_height, x_1)
print(sympy.latex(half_loc[0]))
print(sympy.latex(half_loc[1]))
width = half_loc[1] - half_loc[0]
print(width.simplify())
print(sympy.latex(width.simplify()))

# %% Peak height
loc = -b/(2*a)
equation = a*x_1**2 + b*x_1 + c
height = equation.subs(x_1, loc)
print(height)
print(sympy.latex(height))
