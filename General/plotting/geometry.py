from __future__ import annotations

import math
from collections.abc import Sequence
import functools

import attrs
import sympy
import matplotlib.pyplot as plt
import numpy as np

from General.checking._descriptors import Descriptor


@attrs.define(frozen=True)
class Point:
    x: float
    y: float

    def __getitem__(self, item):
        if item == 0 or item == -2:
            return self.x
        elif item == 1 or item == -1:
            return self.y
        raise IndexError('Index out of range')

    def __iter__(self):
        yield self.x
        yield self.y

    def __add__(self, other: Point | float | int):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        return Point(self.x + other, self.y + other)

    def __radd__(self, other: Point):
        return self.__add__(other)

    def __sub__(self, other: Point | float | int):
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        return Point(self.x - other, self.y - other)

    def __rsub__(self, other: Point | float | int):
        return -1*self.__sub__(other)

    def __mul__(self, other: float | int):
        return Point(self.x*other, self.y*other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return Point(self.x/other, self.y/other)

    def dot(self, other):
        return self.x*other.x + self.y*other.y

    def cross(self, other):
        return self.x*other.y - self.y*other.x

    def close(self, other):
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)


@attrs.define(frozen=True)
class LineSegment:
    point1: Point
    point2: Point

    def intersection(self, other: LineSegment):
        """
        Find the intersection point of the line with another line.

        Parameters
        ----------
        other: LineSegment
            The other line to find the intersection with

        Returns
        -------
        Point | None
            The intersection point or None there is no intersection between the points
        """
        x1, y1 = self.point1
        x2, y2 = self.point2
        x3, y3 = other.point1
        x4, y4 = other.point2

        denominator = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if denominator == 0:
            return None

        t = ((x1 - x3)*(y3 - y4) - (y1 - y3)*(x3 - x4))/denominator
        return Point(x1 + t*(x2 - x1), y1 + t*(y2 - y1))

    def line_formula(self):
        """
        Get the formula of the line in the form of y = mx + b

        Returns
        -------
        Tuple[float, float]
            The slope and y-intercept of the line
        """
        x1, y1 = self.point1
        x2, y2 = self.point2
        m = (y2 - y1)/(x2 - x1)
        b = y1 - m*x1
        return m, b

    def interpolate(self, x, extrapolate=False) -> float:
        """
        Interpolate the y value at a given x value.

        Parameters
        ----------
        x: float
            The x value to interpolate
        extrapolate: bool
            If True, the function will extrapolate the y value if the x value is outside the two points

        Returns
        -------
        float
            The interpolated y value

        Raises
        ------
        ValueError
            If the `x` value is not between the two points

        Notes
        -----
        When the line is vertical, the returned y value is the average of the two points.
        """
        if (not extrapolate) and not (self.point1.x < x < self.point2.x):
            raise ValueError('The `x` value must be between the two points')
        if math.isclose(self.point1.x, self.point2.x):
            return (self.point1.y + self.point2.y)/2
        if math.isclose(self.point1.x, x):
            return self.point1.y
        if math.isclose(self.point2.x, x):
            return self.point2.y

        m, b = self.line_formula()
        return m*x + b

    @staticmethod
    def zeros():
        return LineSegment(Point(0, 0), Point(0, 0))

    def __getitem__(self, item):
        if item == 0 or item == -2:
            return self.point1
        elif item == 1 or item == -1:
            return self.point2
        raise IndexError('Index out of range')

    def __iter__(self):
        yield self.point1
        yield self.point2


@attrs.define(frozen=True)
class Line:
    """
    A line defined by a series of points. The line is defined by the segments between the points. The points must be in order
    of increasing x values.
    """
    points: Sequence[Point]

    def __attrs_post_init__(self):
        if not all(self.points[i].x <= self.points[i + 1].x for i in range(len(self.points) - 1)):
            raise ValueError('The points must be in order of increasing x values')

    @staticmethod
    def from_segments(*segments: LineSegment):
        segments = list(segments)
        output = [segments[0][0]]
        for index in range(len(segments) - 1):
            segment1 = segments[index]
            segment2 = segments[index + 1]
            crossing = segment1.intersection(segment2)
            if crossing is not None:
                output.append(crossing)
                segments[index + 1] = LineSegment(crossing, segment1[1])
            else:
                output.append(segment1[1])
                if not segment1[1].close(segment2[0]):
                    output.append(segment2[0])
                if segment2[0].x < segment1[1].x:
                    raise ValueError(f'The segments {segment1} and {segment2} cannot be connected')
        output.append(segments[-1][1])
        return Line(output)

    @functools.cached_property
    def x_values(self):
        return np.array([point.x for point in self.points])

    @functools.cached_property
    def y_values(self):
        return np.array([point.y for point in self.points])

    def interpolate(self, x_values, extrapolate=False):
        if not extrapolate:
            if not all(self.points[0].x <= x <= self.points[-1].x for x in x_values):
                raise ValueError('The `x_values` must be between the first and last points')
        bounds = self.x_values
        indexes = np.searchsorted(bounds, x_values)
        indexes[indexes == 0] = 1
        indexes[indexes == len(bounds)] = len(bounds) - 1

        dx = np.diff(bounds)
        dy = np.diff(self.y_values)

        return self.y_values[indexes - 1] + dy[indexes - 1]*(x_values - bounds[indexes - 1])/dx[indexes - 1]

    def __len__(self):
        return len(self.points)

    def __getitem__(self, item):
        return self.points[item]

    def __iter__(self):
        yield from self.points

    def points(self):
        return [point for segment in self.points for point in segment]


@attrs.define(frozen=True)
class Circle:
    center: Point
    radius: float = Descriptor.positive_number(False)

    def __contains__(self, point: Point):
        if isinstance(point, Point):
            return math.dist(self.center, point) <= self.radius
        elif isinstance(point, tuple) and len(point) == 2:
            return Point(*point) in self
        raise TypeError(f'Can only check for Point in circle, not {type(point)}')


@attrs.define(frozen=True)
class Ellipse:
    center: Point
    radii: Point

    def __attrs_post_init__(self):
        if self.radii.x <= 0 or self.radii.y <= 0:
            raise ValueError('The radii of the ellipse must be positive')

    def __contains__(self, point: Point):
        if isinstance(point, Point):
            return ((point.x - self.center.x)/self.radii.x)**2 + ((point.y - self.center.y)/self.radii.y)**2 <= 1
        elif isinstance(point, tuple) and len(point) == 2:
            return Point(*point) in self
        raise TypeError(f'Can only check for Point in ellipse, not {type(point)}')


@attrs.define(frozen=True)
class Rectangle:
    center: Point
    width: float = Descriptor.positive_number(False)
    height: float = Descriptor.positive_number(False)

    @staticmethod
    def from_points(point1: Point, point2: Point) -> Rectangle:
        if point1.x == point2.x:
            raise ValueError('The points must have different x coordinates')
        if point1.y == point2.y:
            raise ValueError('The points must have different y coordinates')
        center = (point1 + point2)/2
        width = abs(point1.x - point2.x)
        height = abs(point1.y - point2.y)
        return Rectangle(center, width, height)

    @staticmethod
    def from_corners(corner1: Point, corner2: Point) -> Rectangle:
        center = (corner1 + corner2)/2
        width = abs(corner1.x - corner2.x)
        height = abs(corner1.y - corner2.y)
        return Rectangle(center, width, height)

    @staticmethod
    def from_corner(corner: Point, width: float, height: float) -> Rectangle:
        """
        Create a rectangle from a corner point, width, and height. With positive width and height, the corner is the top-left,
        with negative width and height, the corner is the bottom-right, etc.
        """
        return Rectangle(corner + Point(width/2, -height/2), abs(width), abs(height))

    def corners(self) -> tuple[Point, Point, Point, Point]:
        """
        Get the corners of the rectangle. The order is top-left, top-right, bottom-right, bottom-left.
        """
        return (Point(self.center.x - self.width/2, self.center.y + self.height/2),
                Point(self.center.x + self.width/2, self.center.y + self.height/2),
                Point(self.center.x + self.width/2, self.center.y - self.height/2),
                Point(self.center.x - self.width/2, self.center.y - self.height/2))


def two_circle_tangent(circle1: Circle, circle2: Circle):
    """
    Find the tangent points of two circles.

    Parameters
    ----------
    circle1: Circle
        The first circle
    circle2: Circle
        The second circle

    Returns
    -------
    Tuple[LineSegment, LineSegment]
        The two tangent line segments of the two circles
    """
    distance = math.dist(circle1.center, circle2.center)
    if distance < circle1.radius or distance < circle2.radius:
        # If one of the circles is inside the other, there are no tangent points
        return None, None

    # Find the tangent points, for if the first circle was on the origin and the second on the x-axis
    alpha = math.asin(circle1.radius*(circle2.radius-circle1.radius)/distance)

    point1 = (-circle1.radius*math.sin(alpha), circle1.radius*math.cos(alpha))
    point2 = (distance - circle2.radius*math.sin(alpha), circle2.radius*math.cos(alpha))
    point3 = (-circle1.radius*math.sin(alpha), -circle1.radius*math.cos(alpha))
    point4 = (distance - circle2.radius*math.sin(alpha), -circle2.radius*math.cos(alpha))

    # The angle of the line between the two centres with the x-axis
    rot_angle = math.atan2(circle2.center.y - circle1.center.y, circle2.center.x - circle1.center.x)

    # Rotate the points to their real angles
    r_matrix = [[math.cos(rot_angle), -math.sin(rot_angle)],
                [math.sin(rot_angle), math.cos(rot_angle)]]

    def matrix_mul(matrix, point):
        return matrix[0][0]*point[0] + matrix[0][1]*point[1], matrix[1][0]*point[0] + matrix[1][1]*point[1]

    point1 = Point(*matrix_mul(r_matrix, point1)) + circle1.center
    point2 = Point(*matrix_mul(r_matrix, point2)) + circle1.center
    point3 = Point(*matrix_mul(r_matrix, point3)) + circle1.center
    point4 = Point(*matrix_mul(r_matrix, point4)) + circle1.center

    return LineSegment(point1, point2), LineSegment(point3, point4)


def circle_inside(circle1: Circle, circle2: Circle) -> bool:
    """
    Check if `circle2` is inside `circle1`.

    Parameters
    ----------
    circle1: Circle
        The first circle
    circle2: Circle
        The second circle

    Returns
    -------
    bool
    """
    return math.dist(circle1.center, circle2.center) + circle2.radius <= circle1.radius


def two_ellipse_tangent(ellipse1: Ellipse, ellipse2: Ellipse):
    """
    Find the tangent points of two ellipses. The ellipses are defined by their centres and radii.

    Parameters
    ----------
    ellipse1: Ellipse
        The first ellipse
    ellipse2: Ellipse
        The second ellipse

    Returns
    -------
    Tuple[LineSegment, LineSegment] | None
        The two tangent line segments between the two ellipses, or None if one ellipse is fully inside the other.

    Notes
    ------
    Since the equation of the ellipse is not linear, the solution is found by solving the equation of the tangent line using sympy.
    """
    if ellipse_check_inside(ellipse1, ellipse2) != -1:
        return None

    b, m = sympy.symbols('b m')
    x, y, sx, sy = sympy.symbols('x y sx sy')

    equation = b ** 2 + 2 * b * m * x - 2 * b * y - m ** 2 * sx ** 2 + m ** 2 * x ** 2 - 2 * m * x * y - sy ** 2 + y ** 2
    equation1 = equation.subs([(sx, ellipse1.radii.x), (sy, ellipse1.radii.y), (x, ellipse1.center.x), (y, ellipse1.center.y)])
    equation2 = equation.subs([(sx, ellipse2.radii.x), (sy, ellipse2.radii.y), (x, ellipse2.center.x), (y, ellipse2.center.y)])
    solutions = sympy.solve([equation1, equation2], b, m)

    x_val = (-b*m*sx**2 + m*sx**2*y + sy**2*x)/(m**2*sx**2 + sy**2)
    x1 = x_val.subs([(b, solutions[3][0]), (m, solutions[3][1]), (sx, ellipse1.radii.x), (sy, ellipse1.radii.y),
                     (x, ellipse1.center.x, y, ellipse1.center.y)])
    x2 = x_val.subs([(b, solutions[3][0]), (m, solutions[3][1]), (sx, ellipse2.radii.x),
                     (sy, ellipse2.radii.y), (x, ellipse2.center.x, y, ellipse2.center.y)])

    y_val = b + m*x
    y1 = y_val.subs([(b, solutions[3][0]), (m, solutions[3][1]), (x, x1)])
    y2 = y_val.subs([(b, solutions[3][0]), (m, solutions[3][1]), (x, x2)])

    return (x1, y1), (x2, y2)


def ellipse_check_inside(ellipse1, ellipse2):
    """
    Check if one ellipse is inside the other.

    Parameters
    ----------
    ellipse1: Ellipse
        The first ellipse
    ellipse2: Ellipse
        The second ellipse

    Returns
    -------
    int
        The index of the ellipse that is inside the other, 0 for the first ellipse, 1 for the second ellipse, -1 for neither

    Notes
    -----
    The function will transform the problem to whether an ellipse is inside the unit circle. It will then check if the ellipse is
    inside the unit circle by checking if the points on the ellipse are inside the unit circle. In some cases where one ellipse is
    just outside the unit circle, the function may return -1.
    """
    if ellipse_inside(ellipse1, ellipse2):
        return 1
    elif ellipse_inside(ellipse1, ellipse2):
        return 0
    return -1


def ellipse_inside(ellipse1: Ellipse, ellipse2: Ellipse) -> bool:
    """
    Check if `ellipse2` is inside `ellipse1`.

    Parameters
    ----------
    ellipse1: Ellipse
        The first ellipse
    ellipse2: Ellipse
        The second ellipse

    Returns
    -------
    bool

    Notes
    -----
    The function will transform the problem to whether an ellipse is inside the unit circle. It will then check if the ellipse is
    inside the unit circle by checking if the points on the ellipse are inside the unit circle. In some cases where one ellipse is
    just outside the unit circle, the function may return False.
    """
    # Transform the problem to whether an ellipse is inside the unit circle
    center = Point(ellipse2.center.x - ellipse1.center.x, ellipse2.center.y - ellipse1.center.y)
    radii = Point(ellipse2.radii.x / ellipse1.radii.x, ellipse2.radii.y / ellipse1.radii.y)

    # Check if the ellipse is inside the unit circle
    for angle in range(0, 360, 1):
        x = radii.x * math.cos(math.radians(angle)) + center.x
        y = radii.y * math.sin(math.radians(angle)) + center.y
        if x ** 2 + y ** 2 > 1:
            return False
    return True
