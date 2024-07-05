import sympy

x, y = sympy.symbols('x y')
m, b = sympy.symbols('m b')
sx, sy = sympy.symbols('sx sy')
sx2, sy2 = sympy.symbols('sx2 sy2')

# The equation of the ellipse
tangent = x**2/sx**2 + (m*x+b)**2/sy**2 - 1

x_solution = sympy.solve(tangent, x)  # sx*(-b*m*sx - sy*sqrt(-b**2 + m**2*sx**2 + sy**2))/(m**2*sx**2 + sy**2)

tangent = tangent.subs(x, sx*(-b*m*sx)/(m**2*sx**2 + sy**2))  # Discriminant is 0 since it is a tangent
simple = sympy.simplify(tangent)  # (b**2 - m**2*sx**2 - sy**2)/(m**2*sx**2 + sy**2)

equation1 = b**2 - m**2*sx**2 - sy**2  # The simple equation should be zero, thus the numerator should be zero
equation2 = b**2 - m**2*sx2**2 - sy2**2  # The same equation for the second ellipse

solution = sympy.solve([equation1, equation2], b, m)
print(solution)

print(solution[3][0].subs([(sx, 5), (sy, 3), (sx2, 4), (sy2, 5)]).simplify())

# %%
x, y = sympy.symbols('x y')
m, b = sympy.symbols('m b')
sx, sy = sympy.symbols('sx sy')
sx2, sy2 = sympy.symbols('sx2 sy2')
x1, y1, x2, y2 = sympy.symbols('x1 y1 x2 y2')

# The equation of the ellipse
tangent = (x - x1)**2/sx**2 + ((m*x+b)-y1)**2/sy**2 - 1

x_solution = sympy.solve(tangent, x)  # (-b*m*sx**2 + m*sx**2*y1 - sx*sy*sqrt(-b**2 - 2*b*m*x1 + 2*b*y1 + m**2*sx**2 - m**2*x1**2 + 2*m*x1*y1 + sy**2 - y1**2) + sy**2*x1)/(m**2*sx**2 + sy**2)

tangent = tangent.subs(x, (-b*m*sx**2 + m*sx**2*y1 + sy**2*x1)/(m**2*sx**2 + sy**2))  # Discriminant is 0 since it is a tangent
simple = sympy.simplify(tangent)  # (b**2 + 2*b*m*x1 - 2*b*y1 - m**2*sx**2 + m**2*x1**2 - 2*m*x1*y1 - sy**2 + y1**2)/(m**2*sx**2 + sy**2)

equation1 = b**2 + 2*b*m*x1 - 2*b*y1 - m**2*sx**2 + m**2*x1**2 - 2*m*x1*y1 - sy**2 + y1**2  # The simple equation should be zero, thus the numerator should be zero
equation2 = b**2 + 2*b*m*x2 - 2*b*y2 - m**2*sx2**2 + m**2*x2**2 - 2*m*x2*y2 - sy2**2 + y2**2  # The same equation for the second ellipse

solution = sympy.solve([equation1, equation2], b, m)
print(solution)

print(solution[3][0].subs([(sx, 5), (sy, 3), (sx2, 4), (sy2, 5), (x1, 0), (x2, 0), (y1, 0), (y2, 0)]).simplify())
