
from __future__ import print_function

# Created on 20190206
# Author: Yaoyu Hu <yyhu_live@outlook.com>

import math
import numpy as np

VALID_INTERSECTION    = 0
FALL_OUT_INTERSECTION = 1
PARALLAL              = 2
NO_VALID_INTERSECTION = 3

def is_inside_line_segment(x, y, x0, y0, x1, y1):
    """Return True if the (x, y) lies inside the line segment defined by
    (x0, y0) and (x1, y1)."""

    # Create two vectors.
    v0 = np.array([ x0-x, y0-y ]).reshape((2,1))
    v1 = np.array([ x1-x, y1-y ]).reshape((2,1))

    # Inner product.
    prod = v0.transpose().dot(v1)

    if ( prod <= 0 ):
        return True
    else:
        return False

def line_intersect(x0, y0, x1, y1, x2, y2, x3, y3, eps = 1e-6):
    """
    Calculates the interscetion point of two line segments (x0, y0) - (x1, y1) and
    (x2, y2) - (x3, y3). 
    
    The return values are [x, y] and a flag. flag == VALID_INTERSECTION means a valid intersection 
    point is found. flag == FALL_OUT_INTERSECTION means intersection doese not fall into the line segments. 
    flag == PARALLEL means lines are parallel. flag == NO_VALID_INTERSECTION means other situations, 
    such as degenerated line segments.

    The function will test if these two lines are parallel by
    comparing the differences of x2 - x0 and x3 - x1, y2 - y0 and y3 - y1. If these 
    difference falls with in a range specified by eps, then these two lines are
    considered to be parallel to each other. If parallel lines are detected, this 
    function returns PARALLEL as the value of flag. eps must be a positive number.

    In case of non-parallel lines, the function calculates the intersection point
    with the extended lines of the input line segments. Then the intersection point
    is tested against the extent of these two lines. If the intersection falls into both
    of these line segments, then it is treated as a valid intersection, flag will be VALID_INTERSECTION. 
    Otherwise, flag will be FALL_OUT_INTERSECTION.

    For both the cases of flag == VALID_INTERSECTION and flag == FALL_OUT_INTERSECTION, the
    returned [x, y] contains the intersection point. For other cases, [x, y] is [None, None].
    """

    # Test the order of input arguments.
    assert( eps > 0 )

    # Find out the order of x coordinates
    if ( x0 <= x1 ):
        minX01, maxX01 = x0, x1
    else:
        minX01, maxX01 = x1, x0
    
    if ( x2 <= x3 ):
        minX23, maxX23 = x2, x3
    else:
        minX23, maxX23 = x3, x2

    # Find out the order of y coordinates
    if ( y0 <= y1 ):
        minY01, maxY01 = y0, y1
    else:
        minY01, maxY01 = y1, y0
    
    if ( y2 <= y3 ):
        minY23, maxY23 = y2, y3
    else:
        minY23, maxY23 = y3, y2

    # Initialize the return values.
    x    = None
    y    = None
    flag = NO_VALID_INTERSECTION

    dx0 = x1 - x0
    dy0 = y1 - y0

    dx2 = x3 - x2
    dy2 = y3 - y2

    # Test if lines segments are degenerated.
    d01 = math.sqrt( dx0**2 + dy0**2 )
    d23 = math.sqrt( dx2**2 + dy2**2 )
    if ( d01 < eps ):
        return [x, y], flag
    
    if ( d23 < eps ):
        return [x, y], flag

    # Test if lines are parallel.
    if ( math.fabs( math.fabs( dx0*dx2 + dy0*dy2 ) - d01*d23 ) <= eps ):
        return [x, y], PARALLAL       

    # ========== Intersection calculation. ==========

    # Calculate the intersection.
    a11 = dy0; a12 = -dx0
    a21 = dy2; a22 = -dx2

    AI = np.array( \
        [ [  a22, -a12 ], \
          [ -a21,  a11 ] ],\
        dtype = np.float32 ) / ( a11*a22 - a12*a21 )
    RHS = np.array([ x0*y1 - x1*y0, x2*y3 - x3*y2 ], dtype = np.float32).reshape((2, 1))

    X = AI.dot( RHS )
    x = X[0, 0]; y = X[1, 0]

    # Test if the intersection falls into the line segments.
    # if ( ( ( x >= minX01 and x <= maxX01 ) or ( y >= minY01 and y <= maxY01 ) ) and \
    #      ( ( x >= minX23 and x <= maxX23 ) or ( y >= minY23 and y <= maxY23 ) ) ):
    #     flag = VALID_INTERSECTION
    # else:
    #     flag = FALL_OUT_INTERSECTION

    if ( is_inside_line_segment( x, y, x0, y0, x1, y1 ) and \
         is_inside_line_segment( x, y, x2, y2, x3, y3 ) ):
        flag = VALID_INTERSECTION
    else:
        flag = FALL_OUT_INTERSECTION

    return [ x, y ], flag
