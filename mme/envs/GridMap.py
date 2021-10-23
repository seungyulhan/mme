
from __future__ import print_function

import copy
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn, rand
import os

from mme.envs import LineIntersection2D

def two_point_distance(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0

    return math.sqrt( dx**2 + dy**2 )

def round_if_needed(x, eps = 1e-4):
    """Rount to the nearest integer if x falls into the eps around a integer."""
    
    # if ( isinstance( x, (int, long) ) ):
    #     return x

    if ( x is None ):
        return x

    temp = float(np.ceil(x))
    if ( temp - x < eps ):
        return temp

    temp = float(np.floor(x))
    if ( x - temp < eps ):
        return temp

    return float(x)

class GridMapException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr( self.msg )

class BlockIndex(object):
    def __init__(self, r, c):
        # assert( isinstance(r, (int, long)) )
        # assert( isinstance(c, (int, long)) )
        
        self.r = r
        self.c = c

        self.size = 2
    
    def __str__(self):
        return "index({}, {})".format( self.r, self.c )

class BlockCoor(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.size = 2

    def __str__(self):
        return "coor({}, {})".format( self.x, self.y )

def two_coor_distance(c0, c1):
    dx = c1.x - c0.x
    dy = c1.y - c0.y

    return math.sqrt( dx**2 + dy**2 )

class BlockCoorDelta(object):
    def __init__(self, dx, dy):
        self.dx = dx
        self.dy = dy

        self.size = 2

    def __str__(self):
        return "CoorDelta({}, {})".format( self.dx, self.dy )

    def convert_to_direction_delta(self):
        dx = 0
        dy = 0

        if ( self.dx > 0 ):
            dx = 1.0
        elif ( self.dx < 0 ):
            dx = -1.0

        if ( self.dy > 0 ):
            dy = 1.0
        elif ( self.dy < 0 ):
            dy = -1.0
        
        return BlockCoorDelta( dx, dy )

class Block(object):
    def __init__(self, x = 0, y = 0, h = 1, w = 1):
        # if ( ( not isinstance(x, (int, long)) ) or \
        #      ( not isinstance(y, (int, long)) ) or \
        #      ( not isinstance(h, (int, long)) ) or \
        #      ( not isinstance(w, (int, long)) ) ):
        #     raise TypeError("x, y, h, and w must be integers.")
        
        self.coor = [x, y]
        self.size = [h, w]
        self.corners = [ \
            [0, 0],\
            [0, 0],\
            [0, 0],\
            [0, 0]\
        ]

        self.update_corners()

        self.name  = "Default name"
        self.id    = 0
        self.color = "#FFFFFFFF" # RGBA order.
        self.value = 0

    def update_corners(self):
        x = self.coor[0]
        y = self.coor[1]
        h = self.size[0]
        w = self.size[1]
        
        self.corners = [ \
            [x,   y],\
            [x+w, y],\
            [x+w, y+h],\
            [x,   y+h]\
        ]

    def set_coor(self, x, y, flagUpdate = True):
        # Check the arguments.
        # if ( (not isinstance(x, (int, long))) or (not isinstance(y, (int, long))) ):
        #     raise TypeError("x and y must be integers.")
        
        self.coor = [x, y]

        if ( True == flagUpdate ):
            self.update_corners()
    
    def set_size(self, h, w, flagUpdate = True):
        # Check the arguments.
        # if ( ( not isinstance(h, (int, long)) ) or ( not isinstance(w, (int, long)) ) ):
        #     raise TypeError("h and w must be integers.")

        if ( h <= 0 or w <= 0 ):
            raise ValueError("h and w must be positive values, h = %d, w = %d" % (h, w))
        
        self.size = [h, w]
        if ( True == flagUpdate ):
            self.update_corners()

    def set_coor_size(self, x, y, h, w):
        self.set_coor( x, y, False )
        self.set_size( h, w )
    
    def get_coor(self, idx = 0):
        # Argument check.
        # if ( not isinstance(idx, (int, long)) ):
        #     raise TypeError("idx must be an integer")
        
        if ( idx < 0 or idx > 3 ):
            raise IndexError("idx out of range, idx = %d" % (idx))
        
        return self.corners[idx]

    def is_inside(self, x, y):
        if ( x >= self.corners[0][0] and \
             x <  self.corners[1][0] and \
             y >= self.corners[0][1] and \
             y <  self.corners[3][1] ):
            return True
        
        return False

class NormalBlock(Block):
    def __init__(self, x = 0, y = 0, h = 1, w = 1, value = -0.1):
        super(NormalBlock, self).__init__(x, y, h, w)

        # Member variables defined in the super classes.
        self.color = "#FFFFFFFF"
        self.name  = "NormalBlock"
        self.value = value
    
class ObstacleBlock(Block):
    def __init__(self, x = 0, y = 0, h = 1, w = 1, value = -10):
        super(ObstacleBlock, self).__init__(x, y, h, w)

        # Member variables defined in the super classes.
        self.color = "#FF0000FF"
        self.name  = "ObstacleBlock"
        self.value = value

class StartingBlock(Block):
    def __init__(self, x = 0, y = 0, h = 1, w = 1, value = -0.1, startingPoint=None):
        super(StartingBlock, self).__init__(x, y, h, w)

        # Member variables defined in the super classes.
        self.color = "#00FF00FF"
        self.name  = "StartingBlock"
        self.value = value

        self.startingPoint = [ x + w/2.0, y + h/2.0 ]

        if ( startingPoint is not None ):
            self.set_starting_point( startingPoint[0], startingPoint[1] )

    def set_starting_point(self, x, y):
        if ( self.is_inside(x, y) ):
            self.startingPoint = [ x, y ]
        else:
            raise GridMapException("Specified coordinate (%f, %f) is out of the range of the starting block ( %f <= x < %f, %f <= y < %f )." % \
                x, y, self.corners[0][0], self.corners[1][0], self.corners[0][1], self.corners[3][1])

    def get_starting_point_coor(self):
        coor = BlockCoor( self.startingPoint[0], self.startingPoint[1] )
        return coor
    
    def get_starting_point_list(self):
        return self.startingPoint

class EndingBlock(Block):
    def __init__(self, x = 0, y = 0, h = 1, w = 1, value = 100, endPoint=None):
        super(EndingBlock, self).__init__(x, y, h, w)
    
        # Member variables defined in the super classes.
        self.color = "#0000FFFF"
        self.name  = "EndingBlock"
        self.value = value

        self.endPoint = [x + w/2.0, y + h/2.0]

        if ( endPoint is not None ):
            self.set_end_point( endPoint[0], endPoint[1] )
    
    def set_end_point(self, x, y):
        if ( self.is_inside( x, y ) ):
            self.endPoint = [x, y]
        else:
            raise GridMapException("Specified coordinate (%f, %f) is out of the range of the ending block ( %f <= x < %f, %f <= y < %f )." % \
                x, y, self.corners[0][0], self.corners[1][0], self.corners[0][1], self.corners[3][1])
    
    def is_in_range(self, x, y, r):
        if ( r < 0 ):
            raise GridMapException( "r should not be negative. r = %f." % (r) )

        dx = x - self.endPoint[0]
        dy = y - self.endPoint[1]

        d = math.sqrt( dx**2 + dy**2 )

        if ( d <= r ):
            return True
        else:
            return False

    def get_ending_point_coor(self):
        coor = BlockCoor( self.endPoint[0], self.endPoint[1] )
        return coor
    
    def get_ending_point_list(self):
        return copy.deepcopy( self.endPoint )

def add_element_to_2D_list(ele, li):
    """
    This function tests the existance of ele in list li.
    If ele is not in li, then ele is appended to li.

    li is a 2D list. ele is supposed to have the same number of elements with 
    each element in list li.

    ele: A list.
    li: The targeting 2D list.
    """

    # Test if li is empty.
    nLi  = len( li )

    if ( 0 == nLi ):
        li.append( ele )
        return
    
    # Use nLi again.
    nLi = len( li[0] )

    nEle = len(ele)

    assert( nLi == nEle and nLi != 0 )

    # Try to find ele in li.
    for e in li:
        count = 0

        for i in range(nEle):
            if ( ele[i] == e[i] ):
                count += 1
            
        if ( nEle == count ):
            return
    
    li.append( ele )

class GridMap2D(object):
    I_R = 0
    I_C = 1
    I_X = 0
    I_Y = 1

    def __init__(self, rows, cols, origin = [0, 0], stepSize = [1, 1], name = "", outOfBoundValue = -10):
        # assert( isinstance(rows, (int, long)) )
        # assert( isinstance(cols, (int, long)) )
        # assert( isinstance(origin[0], (int, long)) )
        # assert( isinstance(origin[1], (int, long)) )
        assert( rows > 0 )
        assert( cols > 0 )

        self.isInitialized = False

        self.name      = name # Should be a string.

        self.rows      = rows
        self.cols      = cols
        self.origin    = copy.deepcopy(origin) # x and y coordinates of the starting coordinate.
        self.stepSize  = copy.deepcopy(stepSize) # Step sizes in x and y direction. Note the order of x and y.
        self.outOfBoundValue = outOfBoundValue

        self.valueStartingBlock = -0.1
        self.valueEndingBlock   = 100
        self.valueNormalBlock   = -0.1
        self.valueObstacleBlock = -10

        self.corners   = [] # A 4x2 2D list. Coordinates.
        self.blockRows = [] # A list contains rows of blocks.

        self.centerCoor = BlockCoor(0, 0)
        self.mapSize = [0, 0] # H, W, or, I_R, I_C

        self.haveStartingBlock = False
        self.startingBlockIdx  = BlockIndex(0, 0)
        self.startingPoint     = BlockCoor(0, 0)

        self.haveEndingBlock = False
        self.endingBlockIdx  = BlockIndex(0, 0)
        self.endingPoint     = BlockCoor(0, 0)

        self.obstacleIndices = []

        # Potential value.
        self.havePotentialValue    = False
        self.potentialValuePerStep = 0.1
        self.potentialValueMax     = 0

    def set_value_normal_block(self, val):
        self.valueNormalBlock = val
    
    def set_value_starting_block(self, val):
        self.valueStartingBlock = val

    def set_value_ending_block(self, val):
        self.valueEndingBlock = val
    
    def set_value_obstacle_block(self, val):
        self.valueObstacleBlock = val
    
    def set_value_out_of_boundary(self, val):
        self.outOfBoundValue = val

    def get_center_coor(self):
        return copy.deepcopy( self.centerCoor )
    
    def get_map_size(self):
        return copy.deepcopy( self.mapSize )

    def initialize(self):
        if ( True == self.isInitialized ):
            raise GridMapException("Map already initialized.")

        # Drop current blockRows.
        self.blockRows = []

        # Generate indices for the blocks.
        rs = np.linspace(self.origin[GridMap2D.I_Y], self.origin[GridMap2D.I_Y] + self.rows - 1, self.rows, dtype = np.int)
        cs = np.linspace(self.origin[GridMap2D.I_X], self.origin[GridMap2D.I_X] + self.cols - 1, self.cols, dtype = np.int)
    
        h = self.stepSize[GridMap2D.I_Y]
        w = self.stepSize[GridMap2D.I_X]

        for r in rs:
            temp = []
            for c in cs:
                b = NormalBlock( c*w, r*h, h, w, self.valueNormalBlock )
                temp.append(b)
            
            self.blockRows.append(temp)
        
        # Calcluate the corners.
        self.corners.append( [        cs[0]*w,      rs[0]*h ] )
        self.corners.append( [ (cs[-1] + 1)*w,      rs[0]*h ] )
        self.corners.append( [ (cs[-1] + 1)*w, (rs[-1]+1)*h ] )
        self.corners.append( [        cs[0]*w, (rs[-1]+1)*h ] )

        # Center.
        self.centerCoor = BlockCoor( \
            ( self.corners[1][GridMap2D.I_X] + self.corners[0][GridMap2D.I_X] ) / 2.0, \
            ( self.corners[3][GridMap2D.I_Y] + self.corners[0][GridMap2D.I_Y] ) / 2.0 )

        # Map size.
        self.mapSize = [ \
            self.corners[3][GridMap2D.I_Y] - self.corners[0][GridMap2D.I_Y], \
            self.corners[1][GridMap2D.I_X] - self.corners[0][GridMap2D.I_X] ]
    
    def dump_JSON(self, fn):
        """
        Save the grid map as a JSON file.
        fn: String of filename.
        """

        # Compose a dictionary.
        d = { \
            "name": self.name, \
            "rows": self.rows, \
            "cols": self.cols, \
            "origin": self.origin, \
            "stepSize": self.stepSize, \
            "outOfBoundValue": self.outOfBoundValue, \
            "valueNormalBlock": self.valueNormalBlock, \
            "valueStartingBlock": self.valueStartingBlock, \
            "valueEndingBlock": self.valueEndingBlock, \
            "valueObstacleBlock": self.valueObstacleBlock, \
            "haveStartingBlock": self.haveStartingBlock, \
            "startingBlockIdx": [ self.startingBlockIdx.r, self.startingBlockIdx.c ], \
            "startingPoint": [ self.startingPoint.x, self.startingPoint.y ], \
            "haveEndingBlock": self.haveEndingBlock, \
            "endingBlockIdx": [ self.endingBlockIdx.r, self.endingBlockIdx.c ], \
            "endingPoint": [ self.endingPoint.x, self.endingPoint.y ], \
            "obstacleIndices": self.obstacleIndices
            }
        
        # Open file.
        fp = open( fn, "w" )

        # Save JSON file.
        json.dump( d, fp, indent=4, sort_keys=True )

        fp.close()
    
    def read_JSON(self, fn):
        """
        Read a map from a JSON file. Create all the elements specified in the file.
        fn: String of filename.
        """

        if ( not os.path.isfile(fn) ):
            raise GridMapException("{} does not exist.".format(fn))
        
        fp = open( fn, "r" )

        d = json.load(fp)

        fp.close()

        # Populate member variables by d.
        self.name = d["name"]
        self.rows = d["rows"]
        self.cols = d["cols"]
        self.origin             = d["origin"]
        self.stepSize           = d["stepSize"]
        self.outOfBoundValue    = d["outOfBoundValue"]
        self.valueNormalBlock   = d["valueNormalBlock"]
        self.valueStartingBlock = d["valueStartingBlock"]
        self.valueEndingBlock   = d["valueEndingBlock"]
        self.valueObstacleBlock = d["valueObstacleBlock"]

        self.initialized = False
        self.initialize()

        if ( True == d["haveStartingBlock"] ):
            self.set_starting_block( \
                BlockIndex( \
                    d["startingBlockIdx"][0], d["startingBlockIdx"][1] ),
                startingPoint=BlockCoor( 
                    d["startingPoint"][0], d["startingPoint"][1] ) )
        
        if ( True == d["haveEndingBlock"] ):
            self.set_ending_block( \
                BlockIndex( \
                    d["endingBlockIdx"][0], d["endingBlockIdx"][1] ),
                endPoint=BlockCoor(
                    d["endingPoint"][0], d["endingPoint"][1] ) )

        for obs in d["obstacleIndices"]:
            self.add_obstacle( \
                BlockIndex( \
                    obs[0], obs[1] ) )

    def get_block(self, index):
        if ( isinstance( index, BlockIndex ) ):
            if ( index.r >= self.rows or index.c >= self.cols ):
                raise IndexError( "Index out of range. indx = [%d, %d]" % (index.r, index.c) )
            
            return self.blockRows[index.r][index.c]
        elif ( isinstance( index, (list, tuple) ) ):
            if ( index[GridMap2D.I_R] >= self.rows or \
                 index[GridMap2D.I_C] >= self.cols ):
                raise IndexError( "Index out of range. indx = [%d, %d]" % (index.r, index.c) )
            
            return self.blockRows[ index[GridMap2D.I_R] ][ index[GridMap2D.I_C] ]

    def is_normal_block(self, index):
        b = self.get_block(index)

        return isinstance( b, NormalBlock )

    def is_obstacle_block(self, index):
        b = self.get_block(index)

        return isinstance( b, ObstacleBlock )

    def is_starting_block(self, index):
        b = self.get_block(index)

        return isinstance( b, StartingBlock )

    def is_ending_block(self, index):
        b = self.get_block(index)

        return isinstance( b, EndingBlock )

    def get_step_size(self):
        """[x, y]"""
        return self.stepSize

    def get_index_starting_block(self):
        """Return a copy of the index of the starting block."""
        if ( False == self.haveStartingBlock ):
            raise GridMapException("No staring point set yet.")
        
        return copy.deepcopy( self.startingBlockIdx )
    
    def get_index_ending_block(self):
        """Return a copy of the index of the ending block."""
        if ( False == self.haveEndingBlock ):
            raise GridMapException("No ending block set yet.")
        
        return copy.deepcopy( self.endingBlockIdx )

    def is_in_ending_block(self, coor):
        """Return ture if coor is in the ending block."""

        if ( True == self.is_out_of_or_on_boundary(coor) ):
            return False
        
        loc = self.is_corner_or_principle_line(coor)
        if ( True == loc[0] or True == loc[1] or True == loc[2] ):
            return False

        idx = loc[3]
        if ( True == isinstance( self.blockRows[ idx.r ][ idx.c ], EndingBlock ) ):
            return True
        
        return False

    def is_around_ending_block(self, coor, radius):
        """Return ture if coor is in a circle defined by the center of the ending block."""

        if ( True == self.is_out_of_or_on_boundary(coor) ):
            return False

        # Get the ending block.
        eb = self.get_block(self.endingBlockIdx)

        return eb.is_in_range( coor.x, coor.y, radius )

    def enable_potential_value(self, valMax = None, valPerStep = None):
        if ( valMax is not None ):
            self.potentialValueMax     = valMax
            self.potentialValuePerStep = valPerStep
        
        self.havePotentialValue = True
    
    def disable_potential_value(self):
        self.havePotentialValue = False

    def get_potential_value(self, idxGoal, idx):
        """Return the potential value based on the distance between the indices of idxGaol and idx."""

        # Calculate the index distance.
        dr = idxGoal.r - idx.r
        dc = idxGoal.c - idx.c

        d = math.sqrt( dr**2 + dc**2 )

        return d * self.potentialValuePerStep

    def update_potential_value(self):
        if ( False == self.havePotentialValue ):
            raise GridMapException("Potential value not enabled.")

        # Loop over all the blocks.
        idx = BlockIndex(0, 0)
        for br in self.blockRows:
            idx.c = 0
            for b in br:
                if ( self.is_normal_block( idx ) ):
                    v = self.potentialValueMax - \
                        self.get_potential_value( self.endingBlockIdx, idx )
                    
                    self.get_block( idx ).value += v

                idx.c += 1
            
            idx.r += 1

    def set_starting_block_s(self, r, c, value = None, startingPoint=None):
        # assert( isinstance(r, (int, long)) )
        # assert( isinstance(c, (int, long)) )
        
        if ( True == self.is_ending_block( BlockIndex(r,c) ) ):
            raise GridMapException("The target index for starting block (%d, %d) is already assigned to an ending block." % (r, c))

        if ( True == self.haveStartingBlock ):
            # Get the coordinate of the original starting block.
            cl = self.get_block( self.startingBlockIdx ).corners[0]

            # Overwrite the old staring point with a NormalBlock.
            self.overwrite_block( self.startingBlockIdx.r, self.startingBlockIdx.c, \
                NormalBlock( cl[GridMap2D.I_X], cl[GridMap2D.I_Y], self.stepSize[GridMap2D.I_Y], self.stepSize[GridMap2D.I_X], value=self.valueNormalBlock ) )
        
        # Overwrite a block. Make it to be a starting block.

        if ( value is not None ):
            self.valueStartingBlock = value

        # Coordinate of the new starting block.
        coor = self.convert_to_coordinates( BlockIndex( r, c ) )

        b = StartingBlock( coor.x, coor.y, self.stepSize[GridMap2D.I_Y], self.stepSize[GridMap2D.I_X], value=self.valueStartingBlock, startingPoint=startingPoint )
        self.overwrite_block( r, c, b )
        self.startingBlockIdx.r = r
        self.startingBlockIdx.c = c
        self.startingPoint = b.get_starting_point_coor()

        self.haveStartingBlock = True

    def set_starting_block(self, index, value = None, startingPoint=None):
        if ( isinstance( index, BlockIndex ) ):
            if ( startingPoint is None ):
                self.set_starting_block_s( index.r, index.c, value )
            else:
                self.set_starting_block_s( index.r, index.c, value, startingPoint=[ startingPoint.x, startingPoint.y ] )
        elif ( isinstance( index, (list, tuple) ) ):
            self.set_starting_block_s( index[GridMap2D.I_R], index[GridMap2D.I_C], value, startingPoint=startingPoint )
        else:
            raise TypeError("index should be an object of BlockIndex or a list or a tuple.")

    def random_starting_block(self, value):
        """
        NOTE: If the map has starting or ending blocks, the new randomly
        asssigned starting block will not override them. The new starting block
        will be overriding a normal block.
        """

        while(True):
            # Random the index value of row and column.
            r = int( math.floor( rand() * self.rows ) )
            c = int( math.floor( rand() * self.cols ) )

            idx = BlockIndex( r, c )

            if ( True == self.is_normal_block(idx) ):
                self.set_starting_block(idx, value=value)
                print("Random starting block at [%d, %d]." % (r, c))
                break
            else:
                print("Random starting block at [%d, %d] failed." % (r, c))

    def set_ending_block_s(self, r, c, value=None, endPoint=None):
        """
        endPoint is a two element list containing the ending point coordinates.
        """
        # assert( isinstance(r, (int, long)) )
        # assert( isinstance(c, (int, long)) )

        if ( True == self.is_starting_block( BlockIndex(r,c) ) ):
            raise GridMapException("The target index for ending block (%d, %d) is already assigned to a starting block." % (r, c))
        
        if ( True == self.haveEndingBlock ):
            # Get the coordinate of the original starting block.
            cl = self.get_block( self.endingBlockIdx ).corners[0]

            # Overwrite the old staring point with a NormalBlock.
            self.overwrite_block( self.endingBlockIdx.r, self.endingBlockIdx.c, \
                NormalBlock( cl[GridMap2D.I_X], cl[GridMap2D.I_Y], self.stepSize[GridMap2D.I_Y], self.stepSize[GridMap2D.I_X], value=self.valueNormalBlock ) )
        
        # Overwrite a block. Make it to be a ending block.

        if ( value is not None ):
            self.valueEndingBlock = value

        # Coordinate of the new ending block.
        coor = self.convert_to_coordinates( BlockIndex( r, c ) )

        b = EndingBlock(coor.x, coor.y, self.stepSize[GridMap2D.I_Y], self.stepSize[GridMap2D.I_X], value=self.valueEndingBlock, endPoint=endPoint)

        self.overwrite_block( r, c, b )
        self.endingBlockIdx.r = r
        self.endingBlockIdx.c = c
        self.endingPoint = b.get_ending_point_coor()

        self.haveEndingBlock = True

        if ( True == self.havePotentialValue ):
            self.update_potential_value()
    
    def set_ending_block(self, index, value=None, endPoint=None):
        if ( isinstance( index, BlockIndex ) ):
            if ( endPoint is None ):
                self.set_ending_block_s( index.r, index.c, value )
            else:
                self.set_ending_block_s( index.r, index.c, value, endPoint=[ endPoint.x, endPoint.y ] )
        elif ( isinstance( index, (list, tuple) ) ):
            self.set_ending_block_s( index[GridMap2D.I_R], index[GridMap2D.I_C], value, endPoint=endPoint )
        else:
            raise TypeError("index should be an object of BlockIndex or a list or a tuple.")

    def random_ending_block(self, value):
        """
        NOTE: If the map has starting or ending blocks, the new randomly
        asssigned ending block will not override them. The new ending block
        will be overriding a normal block.
        """

        epShiftX = rand() * self.stepSize[GridMap2D.I_X]
        epShiftY = rand() * self.stepSize[GridMap2D.I_Y]

        while(True):
            # Random the index value of row and column.
            r = int( math.floor( rand() * self.rows ) )
            c = int( math.floor( rand() * self.cols ) )

            idx = BlockIndex( r, c )

            if ( True == self.is_normal_block(idx) ):
                coor = self.convert_to_coordinates(idx)
                coor.x += epShiftX
                coor.y += epShiftY

                self.set_ending_block(idx, value=value, endPoint=coor)
                print("Random ending block at [%d, %d], with ending point at [%f, %f]." % (r, c, coor.x, coor.y))
                break
            else:
                print("Random ending block at [%d, %d] failed." % (r, c))

    def add_obstacle_s(self, r, c, value=None):
        # assert( isinstance(r, (int, long)) )
        # assert( isinstance(c, (int, long)) )

        # Check if the location is a starting block.
        if ( r == self.startingBlockIdx.r and c == self.startingBlockIdx.c ):
            raise IndexError( "Cannot turn a starting block (%d, %d) into obstacle." % (r, c) )
        
        # Check if the location is a ending block.
        if ( r == self.endingBlockIdx.r and c == self.endingBlockIdx.c ):
            raise IndexError( "Cannot turn a ending block (%d, %d) into obstacle." % (r, c) )

        # Check if the destination is already an obstacle.
        if ( isinstance( self.get_block((r, c)), ObstacleBlock ) ):
            return

        if ( value is not None ):
            self.valueObstacleBlock = value

        # Coordinate of the new ending block.
        coor = self.convert_to_coordinates( BlockIndex( r, c ) )

        self.overwrite_block( r, c, \
            ObstacleBlock(coor.x, coor.y, self.stepSize[GridMap2D.I_Y], self.stepSize[GridMap2D.I_X], value=self.valueObstacleBlock ) )

        # Add the indices into self.obstacleIndices 2D list.
        add_element_to_2D_list( [r, c], self.obstacleIndices )

    def add_obstacle(self, index, value=None):
        if ( isinstance( index, BlockIndex ) ):
            self.add_obstacle_s( index.r, index.c, value )
        elif ( isinstance( index, (list, tuple) ) ):
            self.add_obstacle_s( index[GridMap2D.I_R], index[GridMap2D.I_C], value )
        else:
            raise TypeError("index should be an object of BlockIndex or a list or a tuple.")

    def overwrite_block(self, r, c, b):
        """
        r: Row index.
        c: Col index.
        b: The new block.

        b will be assigned to the specified location by a deepcopy.
        The overwritten block does not share the same coordinates with b.
        The coordinates are assigned by the values of r and c.
        """

        assert( r < self.rows )
        assert( c < self.cols )

        temp = copy.deepcopy(b)
        temp.set_coor_size( c, r, self.stepSize[GridMap2D.I_Y], self.stepSize[GridMap2D.I_X] )

        self.blockRows[r][c] = temp

    def get_string_starting_block(self):
        if ( True == self.haveStartingBlock ):
            s = "starting block at [%d, %d], value = %f." % \
                ( self.startingBlockIdx.r, self.startingBlockIdx.c, self.valueStartingBlock )
        else:
            s = "No starting block."

        return s

    def get_string_ending_block(self):
        if ( True == self.haveEndingBlock ):
            s = "ending block at [%d, %d], value = %f." % \
                ( self.endingBlockIdx.r, self.endingBlockIdx.c, self.valueEndingBlock )
        else:
            s = "No ending block."

        return s

    def get_string_obstacles(self):
        n = len( self.obstacleIndices )

        if ( 0 == n ):
            s = "No obstacles."
            return s
        
        s = "%d obstacles:\n" % (n)

        for obs in self.obstacleIndices:
            s += "[%d, %d]\n" % (obs[GridMap2D.I_R], obs[GridMap2D.I_C])
        
        s += "Value of the last added obstacle block is %f." % (self.valueObstacleBlock)

        return s

    def get_string_corners(self):
        s = "Corners:\n"
        
        for c in self.corners:
            s += "[%f, %f]\n" % ( c[GridMap2D.I_X], c[GridMap2D.I_Y] )

        return s

    def __str__(self):
        title = "GridMap2D \"%s\"." % (self.name)

        strDimensions = \
"""r = %d, c = %d.
origin = [%d, %d], size = [%d, %d].""" \
% (self.rows, self.cols, self.origin[GridMap2D.I_X], self.origin[GridMap2D.I_Y], self.stepSize[GridMap2D.I_X], self.stepSize[GridMap2D.I_Y])

        # Normal block.
        strNormalBlock = "Value of the normal block is %f." % (self.valueNormalBlock)

        # Get the string for staring point.
        strStartingBlock = self.get_string_starting_block()

        # Get the string for ending block.
        strEndingBlock = self.get_string_ending_block()

        # Get the string for obstacles.
        strObstacles = self.get_string_obstacles()

        # Get the string for the corners.
        strCorners = self.get_string_corners()

        s = "%s\n%s\n%s\n%s\n%s\n%s\n%s\n" \
            % ( title, strDimensions, strNormalBlock, strStartingBlock, strEndingBlock, strObstacles, strCorners )

        return s

    def is_out_of_or_on_boundary_s(self, x, y):
        if ( x <= self.corners[0][GridMap2D.I_X] or \
             x >= self.corners[1][GridMap2D.I_X] or \
             y <= self.corners[0][GridMap2D.I_Y] or \
             y >= self.corners[3][GridMap2D.I_Y] ):
            return True
        
        return False

    def is_out_of_or_on_boundary(self, coor):
        """Overloaded function. Vary only in the argument list."""

        if ( isinstance(coor, BlockCoor) ):
            return self.is_out_of_or_on_boundary_s( coor.x, coor.y )
        elif ( isinstance(coor, (list, tuple)) ):
            return self.is_out_of_or_on_boundary_s( coor[GridMap2D.I_X], coor[GridMap2D.I_Y] )
        else:
            raise GridMapException("coor should be either an object of BlockCoor or a list")

    def is_out_of_boundary_s(self, x, y):
        if ( x < self.corners[0][GridMap2D.I_X] or \
             x > self.corners[1][GridMap2D.I_X] or \
             y < self.corners[0][GridMap2D.I_Y] or \
             y > self.corners[3][GridMap2D.I_Y] ):
            return True
        
        return False

    def is_out_of_boundary(self, coor):
        """Overloaded function. Vary only in the argument list."""

        if ( isinstance(coor, BlockCoor) ):
            return self.is_out_of_boundary_s( coor.x, coor.y )
        elif ( isinstance(coor, (list, tuple)) ):
            return self.is_out_of_boundary_s( coor[GridMap2D.I_X], coor[GridMap2D.I_Y] )
        else:
            raise GridMapException("coor should be either an object of BlockCoor or a list")

    def get_index_by_coordinates_s(self, x, y):
        """
        It is assumed that (x, y) is inside the map boundaries.
        x and y are real values.
        A list of two elements is returned. The values inside the returned
        list is the row and column indices of the block.
        """

        c = int( ( 1.0*x - self.origin[GridMap2D.I_X] ) / self.stepSize[GridMap2D.I_X] )
        r = int( ( 1.0*y - self.origin[GridMap2D.I_Y] ) / self.stepSize[GridMap2D.I_Y] )

        return BlockIndex(r, c)

    def get_index_by_coordinates(self, coor):
        """Overloaded funcion. Only varys in the argument list."""

        if ( isinstance(coor, BlockCoor) ):
            return self.get_index_by_coordinates_s( coor.x, coor.y )
        elif ( isinstance(coor, (list, tuple)) ):
            return self.get_index_by_coordinates_s( coor[GridMap2D.I_X], coor[GridMap2D.I_Y])
        else:
            raise TypeError("coor should be either an object of BlcokCoor or a list")

    def sum_block_values(self, idxList):
        """
        Sum the values according to the index list in idxList.

        It is processed as follows:
        * If neighboring block is out of boundary, an outOfBoundaryValue will be added.
        * If neighboring block is an obstacle, the value of an obstacle well be added.
        * If no neighboring block is either out of boundary or an obstacle, only one normal block will be counted.
        """

        # Number of indices.
        n = len( idxList )

        if ( 0 == n ):
            raise GridMapException("The length of idxList must not be zero.")

        if ( 1 == n ):
            idx = idxList[0]
            return self.blockRows[ idx.r ][ idx.c ].value

        flagHaveNormalBlock    = False
        flagHaveNonNormalBlock = False

        val   = 0 # The final value.
        valNB = 0 # The value of the normal block.
        valOB = 0 # The value for out of boundary.

        for idx in idxList:
            # Check if idx is out of boundary.
            if ( idx.r >= self.rows or \
                 idx.c >= self.cols or \
                 idx.r < 0 or \
                 idx.c < 0 ):
                valOB = self.outOfBoundValue
                flagHaveNonNormalBlock = True
                continue

            # Get the actual block.
            b = self.blockRows[ idx.r ][ idx.c ]

            # Check if idx is a normal block.
            if ( isinstance( b, NormalBlock ) ):
                flagHaveNormalBlock = True
                valNB = b.value
                continue

            # Check if idx is an obstacle.
            if ( isinstance( b, ObstacleBlock ) ):
                val += b.value
                flagHaveNonNormalBlock = True
                continue

        # Only count out-of-boundary condition for onece.
        val += valOB

        # Check if all types of blocks are handled.
        if ( False == flagHaveNonNormalBlock and \
             False == flagHaveNormalBlock ):
            raise GridMapException("No blocks are recognized!")

        if ( False == flagHaveNonNormalBlock and \
             True  == flagHaveNormalBlock ):
             # This if condition seems to be unnessesary.
            val += valNB
        
        return val

    def evaluate_coordinate_s(self, x, y):
        """
        This function returns the value coresponds to coor. The rules are as follows:
        (1) If coor is out of boundary but not exactly on the boundary, an exception will be raised.
        (2) If coor is sitting on a block corner, the values from the neighboring 4 blocks will be summed.
        (3) If coor is sitting on a horizontal or vertical line but not a block corner, the neighboring 2 blocks will be summed.
        (4) If coor is inside a block, the value of that block will be returned.

        For summation, it is processed as follows:
        * If neighboring block is out of boundary, an outOfBoundaryValue will be added.
        * If neighboring block is an obstacle, the value of an obstacle well be added.
        * If no neighboring block is either out of boundary or an obstacle, only one normal block will be counted.
        """

        # Check if (x, y) is out of boundary.
        if ( True == self.is_out_of_boundary_s( x, y ) ):
            raise GridMapException("Coordinate (%f, %f) out of boundary. Could not evaluate its value." % ( x, y ))
        
        # In or on the boundary.
        
        # Check if the coordinate is a corner, horizontal, or vertical line of the map.
        loc = self.is_corner_or_principle_line( BlockCoor( x, y ) )

        idxList = [] # The index list of neighoring blocks.
        idx = copy.deepcopy( loc[3] )

        if ( True == loc[0] ):
            # A corner.
            idxList.append( copy.deepcopy(idx) ); idx.c -= 1
            idxList.append( copy.deepcopy(idx) ); idx.r -= 1
            idxList.append( copy.deepcopy(idx) ); idx.c += 1
            idxList.append( copy.deepcopy(idx) )
        elif ( True == loc[1] ):
            # A horizontal line.
            idxList.append( copy.deepcopy(idx) ); idx.r -= 1
            idxList.append( copy.deepcopy(idx) )
        elif ( True == loc[2] ):
            # A vertical line.
            idxList.append( copy.deepcopy(idx) ); idx.c -= 1
            idxList.append( copy.deepcopy(idx) )
        else:
            # A normal block.
            idxList.append( idx )

        # Summation routine.
        val = self.sum_block_values( idxList )

        return val

    def evaluate_coordinate(self, coor):
        """Overloaded function. Only varys in argument list."""

        if ( isinstance( coor, BlockCoor ) ):
            return self.evaluate_coordinate_s( coor.x, coor.y )
        elif ( isinstance( coor, (list, tuple) ) ):
            return self.evaluate_coordinate_s( coor[GridMap2D.I_X], coor[GridMap2D.I_Y] )
        else:
            raise TypeError("coor should be either an object of BlockCoor or a list.")
    
    def convert_to_coordinates_s(self, r, c):
        """Convert the index into the real valued coordinates."""

        # Check if [r, c] is valid.
        # assert( isinstance( r, (int, long) ) )
        # assert( isinstance( c, (int, long) ) )
        # assert( r >= 0 and r < self.rows )
        # assert( c >= 0 and c < self.cols )

        return BlockCoor( c*self.stepSize[GridMap2D.I_X], r*self.stepSize[GridMap2D.I_Y] )

    def convert_to_coordinates(self, index):
        """
        Overloaded function. Only various in the argument list.
        """

        if ( isinstance(index, BlockIndex) ):
            return self.convert_to_coordinates_s( index.r, index.c )
        elif ( isinstance(index, (list, tuple)) ):
            return self.convert_to_coordinates_s( index[GridMap2D.I_R], index[GridMap2D.I_C] )
        else:
            raise TypeError("index must be either an ojbect of BlockIndex or a list.")

    def is_east_boundary(self, coor, eps = 1e-6):
        """Return True if coordinate x lies on the east boundary of the map."""

        assert( eps >= 0 )

        if ( 0 == eps ):
            return ( coor.x == self.corners[1][0] )
        else:
            return ( math.fabs( coor.x - self.corners[1][0] ) < eps )

    def is_north_boundary(self, coor, eps = 1e-6):
        """Return True if coordinate y lies on the north boundary of the map."""

        assert( eps >= 0 )

        if ( 0 == eps ):
            return ( coor.y == self.corners[2][1] )
        else:
            return ( math.fabs( coor.y - self.corners[2][1] ) < eps )
    
    def is_west_boundary(self, coor, eps = 1e-6):
        """Return True if coordinate x lies on the west boundary of the map."""

        assert( eps >= 0 )

        if ( 0 == eps ):
            return ( coor.x == self.corners[0][0] )
        else:
            return ( math.fabs( coor.x - self.corners[0][0] ) < eps )

    def is_south_boundary(self, coor, eps = 1e-6):
        """Return True if coordinate y lies on the south boundary of the map."""

        assert( eps >= 0 )

        if ( 0 == eps ):
            return ( coor.y == self.corners[0][1] )
        else:
            return ( math.fabs( coor.y - self.corners[0][1] ) < eps )

    def is_corner_or_principle_line(self, coor):
        """
        It is NOT rerquired that coor is inside the map.

        The return value contains 4 parts:
        (1) Ture if coor is precisely a corner.
        (2) Ture if coor lies on a horizontal principle line or is a corner.
        (3) Ture if coor lies on a vertical principle line or is a corner.
        (4) A BlockIndex object associated with coor.
        """
        
        if ( coor.x is None or coor.y is None ):
            return False, False, False, None

        # Get the index of (x, y).
        index = self.get_index_by_coordinates(coor)

        # Convert back to coordnates.
        coor2 = self.convert_to_coordinates(index)

        res = [ False, coor.y == coor2.y, coor.x == coor2.x, index ]

        res[0] = (res[1] == True) and (res[2] == True)

        return res

class GridMapEnv(object):
    END_POINT_MODE_BLOCK  = 1
    END_POINT_MODE_RADIUS = 2

    def __init__(self, name = "DefaultGridMapEnv", gridMap = None, workingDir = "./"):
        self.name = name
        self.map  = gridMap
        self.workingDir = workingDir
        self.renderDir = os.path.join( self.workingDir, "Render" )

        self.agentStartingLoc = None # Should be an object of BlockCoor.

        # Ending point mode.
        self.endPointMode = GridMapEnv.END_POINT_MODE_BLOCK
        self.endPointRadius = 1.0 # Should be updated if self.map is set later.

        self.isTerminated = False
        self.nSteps = 0
        self.maxSteps = 0 # Set 0 for no maximum steps.

        self.agentCurrentLoc = copy.deepcopy( self.agentStartingLoc )
        self.agentCurrentAct = None # Should be an object of BlockCoorDelta.

        self.agentLocs = [ copy.deepcopy(self.agentCurrentLoc) ]
        self.agentActs = [ ] # Should be a list of objects of BlockCoorDelta.

        self.totalValue = 0

        self.visAgentRadius    = 1.0
        self.visPathArrowWidth = 1.0

        self.visIsForcePause   = False
        self.visForcePauseTime = 1

        self.tryMoveMaxCount   = 0 # 0 for disabled.

        self.nondimensionalStep = False
        self.nondimensionalStepRatio = 0.25 # For non-dimensional step, the maximum size of individual step compared to the length of the map.
        self.actStepSize = [0, 0] # Two element list. dx and dy.

        self.flagActionClip = True # False
        self.actionClip = [-1, 1]

        self.normalizedCoordinate = False
        self.centerCoordinate     = BlockCoor(0, 0)
        self.halfMapSize          = [1, 1]

        self.isRandomCoordinating = False # If True, a noise will be added to the final coordinate produced by each calling to step() function.
        self.randomCoordinatingVariance = 0 # The variance of the randomized coordinate.

        self.flagActionValue = False
        self.actionValueFactor = 1.0

        self.fig = None # The matplotlib figure.
        self.drawnAgentLocations = 0 # The number of agent locations that have been drawn on the canvas.
        self.drawnAgentPaths     = 0 # The number of agent path that have been drawn on the canvas.

    def set_working_dir(self, workingDir):
        self.workingDir = workingDir
        self.renderDir  = os.path.join( self.workingDir, "Render" )

    def enable_ending_point_radius(self, r):
        assert( r > 0 )

        self.endPointRadius = r
        self.endPointMode = GridMapEnv.END_POINT_MODE_RADIUS

    def disable_ending_point_radius(self):
        self.endPointMode = GridMapEnv.END_POINT_MODE_BLOCK

    def get_ending_point_radius(self):
        return self.endPointRadius

    def check_ending_point_radius(self):
        """
        Check if the starting point's center is in the range of the ending point.
        """

        if ( GridMapEnv.END_POINT_MODE_RADIUS != self.endPointMode ):
            raise GridMapException("Check ending point radius could only be done with the END_POINT_MODE_RADIUS mode")

        if ( False == self.map.haveEndingBlock ):
            raise GridMapException("Map of the environment does not have a ending block.")

        if ( False == self.map.haveStartingBlock ):
            raise GridMapException("Map of the environemnt does not have a starting block.")
        
        # Get the ending point.
        idxEnd = self.map.get_index_ending_block()
        bEnd   = self.map.get_block( idxEnd )
        ep     = bEnd.get_ending_point_coor()

        # Get the starting point.
        idxStart = self.map.get_index_starting_block()
        bStart   = self.map.get_block( idxStart )
        sp       = bStart.get_starting_point_coor()

        # Test the distance.
        d = two_coor_distance( ep, sp )
        
        if ( d <= self.endPointRadius ):
            return False
        else:
            return True

    def random_staring_and_ending_blocks(self):
        if ( self.map is None ):
            raise GridMapException("Could not randomize the starting and ending blocks. self.map is None.")
        
        self.map.random_starting_block( self.map.valueStartingBlock )
        self.map.random_ending_block( self.map.valueEndingBlock )

    def set_max_steps(self, m):
        # assert( isinstance( m, (int, long) ) )
        assert( m >= 0 )

        self.maxSteps = m

    def get_max_steps(self):
        return self.maxSteps

    def get_state_size(self):
        return self.agentCurrentLoc.size
    
    def get_action_size(self):
        return self.agentCurrentAct.size
    
    def is_terminated(self):
        return self.isTerminated

    def enable_force_pause(self, t):
        """
        t: Seconds for pause. Must be positive integer.
        """

        # assert( isinstance( t, (int, long) ) )
        assert( t > 0 )

        self.visForcePauseTime = t
        self.visIsForcePause   = True
    
    def disable_force_pause(self):
        self.visIsForcePause = False

    def enable_nondimensional_step(self):
        if ( self.map is None ):
            raise GridMapException("GridMapEnv could not enable non-dimensional step. self.map is None.")
        
        # self.actStepSize[0] = self.nondimensionalStepRatio * \
        #     ( self.map.corners[1][GridMap2D.I_X] - self.map.corners[0][GridMap2D.I_X] )
        # self.actStepSize[1] = self.nondimensionalStepRatio * \
        #     ( self.map.corners[3][GridMap2D.I_Y] - self.map.corners[0][GridMap2D.I_Y] )

        self.actStepSize[0] = self.map.get_step_size()[ GridMap2D.I_X ]
        self.actStepSize[1] = self.map.get_step_size()[ GridMap2D.I_Y ]

        self.nondimensionalStep = True

    def disable_nondimensional_step(self):
        self.nondimensionalStep = False

    def enable_action_clipping(self, cpMin, cpMax):
        assert( cpMin < cpMax )

        self.actionClip = [ cpMin, cpMax ]
        self.flagActionClip = True

    def disable_action_clipping(self):
        self.flagActionClip = False

    def clip_action(self, action):
        if ( False == self.flagActionClip ):
            raise GridMapException("Action clipping is disabled.")
        
        clipped = copy.deepcopy(action)

        if ( action.dx < self.actionClip[0] ):
            clipped.dx = self.actionClip[0]
        elif ( action.dx > self.actionClip[1] ):
            clipped.dx = self.actionClip[1]

        if ( action.dy < self.actionClip[0] ):
            clipped.dy = self.actionClip[0]
        elif ( action.dy > self.actionClip[1] ):
            clipped.dy = self.actionClip[1]
        
        return clipped

    def enable_normalized_coordinate(self):
        if ( self.map is None ):
            raise GridMapException("GridMapEnv could not enable normalized coordinate. self.map is None.")
        
        self.centerCoordinate = self.map.get_center_coor()
        self.halfMapSize = self.map.get_map_size()
        self.halfMapSize[0] /= 2.0 # H.
        self.halfMapSize[1] /= 2.0 # W.
        
        self.normalizedCoordinate = True
    
    def disable_normalized_coordinate(self):
        self.normalizedCoordinate = False

    def make_a_coor(self, x, y):
        b = BlockCoor(x, y)

        if ( True == self.normalizedCoordinate ):
            b.x = x * self.halfMapSize[GridMap2D.I_C] + self.centerCoordinate.x
            b.y = y * self.halfMapSize[GridMap2D.I_R] + self.centerCoordinate.y
        
        return b

    def enable_random_coordinating(self, v):
        self.isRandomCoordinating = True
        self.randomCoordinatingVariance = v

    def disable_random_coordinating(self):
        self.isRandomCoordinating = False

    def randomize_action(self, coor, action):
        """Randomize the action."""

        # The original target.
        ot = BlockCoor( coor.x + action.dx, coor.y + action.dy )

        # Randomize ot.
        ot.x += self.randomCoordinatingVariance * math.fabs( action.dx ) * randn()
        ot.y += self.randomCoordinatingVariance * math.fabs( action.dy ) * randn()

        # New action.
        return BlockCoorDelta( ot.x - coor.x, ot.y - coor.y )

    def enable_action_value(self, factor):
        assert( factor > 0 )

        self.actionValueFactor = factor
        self.flagActionValue = True
        self.enable_nondimensional_step()

    def disable_action_value(self):
        self.flagActionValue = False
        self.disable_nondimensional_step()

    def reset(self):
        """Reset the evironment."""

        if ( self.map is None ):
            raise GridMapException("Map is None.")

        if ( not os.path.isdir( self.workingDir ) ):
            os.makedirs( self.workingDir )

        if ( not os.path.isdir( self.renderDir ) ):
            os.makedirs( self.renderDir )

        # Close render, if there are any.
        self.close_render()
        self.drawnAgentLocations = 0
        self.drawnAgentPaths     = 0

        # Get the index of the starting block.
        index = self.map.get_index_starting_block()
        # Get the coordinates of index.
        coor = self.map.convert_to_coordinates(index)

        sizeW = self.map.get_step_size()[GridMap2D.I_X]
        sizeH = self.map.get_step_size()[GridMap2D.I_Y]

        self.agentStartingLoc = BlockCoor( \
            coor.x + sizeW / 2.0, \
            coor.y + sizeH / 2.0 \
        )
        
        # Reset the location of the agent.
        self.agentCurrentLoc = copy.deepcopy( self.agentStartingLoc )

        # Clear the cuurent action of the agent.
        self.agentCurrentAct = BlockCoorDelta( 0, 0 )

        # Clear the history.
        self.agentLocs = [ copy.deepcopy( self.agentStartingLoc ) ]
        self.agentActs = [ ]

        # Non-dimensional step size.
        if ( True == self.nondimensionalStep ):
            self.actStepSize[0] = self.nondimensionalStepRatio * \
                ( self.map.corners[1][GridMap2D.I_X] - self.map.corners[0][GridMap2D.I_X] )
            self.actStepSize[1] = self.nondimensionalStepRatio * \
                ( self.map.corners[3][GridMap2D.I_Y] - self.map.corners[0][GridMap2D.I_Y] )

        if ( True == self.normalizedCoordinate ):
            self.centerCoordinate = self.map.get_center_coor()
            self.halfMapSize = self.map.get_map_size()
            self.halfMapSize[0] /= 2.0 # H.
            self.halfMapSize[1] /= 2.0 # W.

        # Clear step counter.
        self.nSteps = 0

        # Clear total value.
        self.totalValue = 0

        # Clear termination flag.
        self.isTerminated = False

        # Visulization.
        if ( sizeW <= sizeH ):
            self.visAgentRadius    = sizeW / 10.0
            self.visPathArrowWidth = sizeW / 10.0
        else:
            self.visAgentRadius    = sizeH / 10.0
            self.visPathArrowWidth = sizeW / 10.0

        # Monitor.
        self.tryMoveMaxCount = max( self.map.rows, self.map.cols ) * 2

        agentCurrentLocation = copy.deepcopy( self.agentCurrentLoc )

        if ( True == self.normalizedCoordinate ):
            agentCurrentLocation.x = ( agentCurrentLocation.x - self.centerCoordinate.x ) / self.halfMapSize[GridMap2D.I_C]
            agentCurrentLocation.y = ( agentCurrentLocation.y - self.centerCoordinate.y ) / self.halfMapSize[GridMap2D.I_R]

        return agentCurrentLocation

    def step(self, action):
        """
        Return values are next state, reward value, termination flag, and None.
        action: An object of BlockCoorDelta.

        action will be deepcopied.
        """

        if ( True == self.isTerminated ):
            print(self.agentCurrentLoc, action, self.nSteps)
            raise GridMapException("Episode already terminated.")
        
        self.agentCurrentAct = copy.deepcopy( action )
        # import ipdb; ipdb.set_trace()
        # Action clipping.
        if ( True == self.flagActionClip ):
            self.agentCurrentAct = self.clip_action( self.agentCurrentAct )

        # Non-dimensional step.
        if ( True == self.nondimensionalStep ):
            self.agentCurrentAct.dx *= self.actStepSize[GridMap2D.I_X]
            self.agentCurrentAct.dy *= self.actStepSize[GridMap2D.I_Y]

        # Random coordinating.
        if ( True == self.isRandomCoordinating ):
            self.agentCurrentAct = self.randomize_action(self.agentCurrentLoc, self.agentCurrentAct)

        # Move.
        newLoc, value, termFlag = self.try_move( self.agentCurrentLoc, self.agentCurrentAct )

        # Additional action value.
        if ( True == self.flagActionValue ):
            value -= self.actionValueFactor * ( max( action.dx**2 + action.dy**2 - 1.0**2 , 0.0 ))

        # Update current location of the agent.
        self.agentCurrentLoc = copy.deepcopy( newLoc )

        # Save the history.
        self.agentLocs.append( copy.deepcopy( self.agentCurrentLoc ) )
        self.agentActs.append( copy.deepcopy( self.agentCurrentAct ) )

        # Update counter.
        self.nSteps += 1

        # Update total value.
        self.totalValue += value

        # Check termination status.
        if ( self.maxSteps > 0 ):
            if ( self.nSteps >= self.maxSteps ):
                self.isTerminated = True
                termFlag = True

        if ( True == termFlag ):
            self.isTerminated = True

        if ( True == self.normalizedCoordinate ):
            newLoc.x = ( newLoc.x - self.centerCoordinate.x ) / self.halfMapSize[GridMap2D.I_C]
            newLoc.y = ( newLoc.y - self.centerCoordinate.y ) / self.halfMapSize[GridMap2D.I_R]
        return newLoc, value, termFlag, {}

    def render(self, pause = 0.0001, flagSave = False, fn = None):
        """Render with matplotlib.
        pause: Time measured in seconds to pause before close the rendered image.
        If pause <= 0 then the rendered image will not be closed and the process
        will be blocked.
        flagSave: Save the rendered image if is set True.
        fn: Filename. If fn is None, a default name scheme will be used. No format extension.

        NOTE: fn should not contain absolute path since the rendered file will be saved
        under the render directory as a part of the working directory.
        """

        if ( self.map is None ):
            raise GridMapException("self.map is None")

        from matplotlib.patches import Rectangle, Circle

        if ( self.fig is None ):
            self.fig, ax = plt.subplots(1)

            for br in self.map.blockRows:
                for b in br:
                    if ( GridMapEnv.END_POINT_MODE_BLOCK == self.endPointMode ):
                        rect = Rectangle( (b.coor[0], b.coor[1]), b.size[1], b.size[0], fill = True)
                        rect.set_facecolor(b.color)
                        # rect.set_edgecolor("k")
                        ax.add_patch(rect)
                    elif ( GridMapEnv.END_POINT_MODE_RADIUS == self.endPointMode ):
                        if ( not isinstance( b, EndingBlock ) ):
                            rect = Rectangle( (b.coor[0], b.coor[1]), b.size[1], b.size[0], fill = True)
                            rect.set_facecolor(b.color)
                            # rect.set_edgecolor("k")
                            ax.add_patch(rect)
                        else:
                            cir = Circle( (b.endPoint[0], b.endPoint[1]), self.endPointRadius, fill=True )
                            cir.set_facecolor(b.color)
                            # cir.set_edgecolor("k")
                    else:
                        raise GridMapException("Unexpected ending point mode %d." % (self.endPointMode))

            if ( GridMapEnv.END_POINT_MODE_RADIUS == self.endPointMode ):
                ax.add_patch(cir)

            # Annotations.
            plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
            plt.xlabel("x", fontsize=12)
            plt.ylabel("y", fontsize=12)
            titleStr = "%s%s" % (self.name, self.map.name)
            plt.title(titleStr)

            # ax.autoscale()
        else:
            ax = plt.gca()
        
        # Agent locations.
        nLocs = len( self.agentLocs )
        if ( nLocs > 0 and self.drawnAgentLocations < nLocs ):
            for i in range( self.drawnAgentLocations, nLocs ):
                circle = Circle( (self.agentLocs[i].x, self.agentLocs[i].y), self.visAgentRadius, fill = True )
                circle.set_facecolor( "#FFFF0080" )
                circle.set_edgecolor( "k" )
                ax.add_patch(circle)
            
            self.drawnAgentLocations = nLocs
        
        # for loc in self.agentLocs:
        #     circle = Circle( (loc.x, loc.y), self.visAgentRadius, fill = True )
        #     circle.set_facecolor( "#FFFF0080" )
        #     circle.set_edgecolor( "k" )
        #     ax.add_patch(circle)

        # Agent path.
        if ( nLocs > 1 and self.drawnAgentPaths < nLocs - 1):
            for i in range(self.drawnAgentPaths, nLocs-1):
                loc0 = self.agentLocs[i]
                loc1 = self.agentLocs[i+1]

                if ( loc0.x == loc1.x and loc0.y == loc1.y ):
                    continue

                plt.arrow( loc0.x, loc0.y, loc1.x - loc0.x, loc1.y - loc0.y, \
                    width=self.visPathArrowWidth, \
                    alpha=0.5, color='k', length_includes_head=True )
                
            self.drawnAgentPaths = nLocs - 1

        plt.xlim( ( self.map.corners[0][GridMap2D.I_X], self.map.corners[1][GridMap2D.I_X] ) )
        plt.ylim( ( self.map.corners[0][GridMap2D.I_Y], self.map.corners[3][GridMap2D.I_Y] ) )

        if ( True == flagSave ):
            if ( fn is None ):
                saveFn = "%s/%s_%d-%ds_%dv.png" % (self.renderDir, self.name, self.nSteps, self.maxSteps, self.totalValue)
            else:
                saveFn = "%s/%s" % (self.renderDir, fn)

            plt.savefig( saveFn, dpi = 300, format = "png" )

        if ( True == self.visIsForcePause ):
            # plt.show( block=False )
            # plt.pause( self.visForcePauseTime )
            # plt.close()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(self.visForcePauseTime)
        else:
            if ( pause <= 0 ):
                plt.show()
            elif ( pause > 0 ):
                print("Render %s for %f seconds." % (self.name, pause))
                # plt.show( block = False )
                # plt.pause( pause )
                # plt.close()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.pause(pause)

    def close_render(self):
        if ( self.fig is not None ):
            plt.close()
            self.fig = None

    def save_render(self, fn):
        if ( self.fig is None ):
            raise GridMapException("No matplotlib figure is present. Could not save figure.")

        plt.savefig( fn, dpi = 300, format = "png" )

    def finalize(self):
        # Close the render.
        self.close_render()

    def save(self, fn = None):
        """
        Save the environment into the working directory.

        If fn == None, a file with the name of GridMapEnv.json will be
        saved into the workding directory.

        fn will be used to create file in the working directory.
        """

        if ( fn is None ):
            fn = "GridMapEnv.json"
                
        fnPart = os.path.splitext(os.path.split(fn)[1])[0]

        strFn  = "%s/%s" % ( self.workingDir, fn )
        mapRef = fnPart + "_Map.json"
        mapFn  = "%s/%s" % ( self.workingDir, mapRef )

        # Check if the map is present.
        if ( self.map is None ):
            raise GridMapException("Map must be set in order to save the environment.")

        # Save the map.
        self.map.dump_JSON( mapFn )

        # Create list for agent location history.
        agentLocsList = []
        for loc in self.agentLocs:
            agentLocsList.append( [loc.x, loc.y] )

        # Create list for agent action history.
        agentActsList = []
        for act in self.agentActs:
            agentActsList.append( [act.dx, act.dy] )

        # Compose a dictionary.
        d = { \
            "name": self.name, \
            "mapFn": mapRef, \
            "maxSteps": self.maxSteps, \
            "nondimensionalStep": self.nondimensionalStep, \
            "nondimensionalStepRatio": self.nondimensionalStepRatio, \
            "flagActionClip": self.flagActionClip, \
            "actionClip": self.actionClip, \
            "flagActionValue": self.flagActionValue, \
            "actionValueFactor": self.actionValueFactor, \
            "endPointMode": self.endPointMode, \
            "endPointRadius": self.endPointRadius, \
            "normalizedCoordinate": self.normalizedCoordinate, \
            "isRandomCoordinating": self.isRandomCoordinating, \
            "randomCoordinatingVariance": self.randomCoordinatingVariance, \
            "actStepSize": self.actStepSize, \
            "visAgentRadius": self.visAgentRadius, \
            "visPathArrowWidth": self.visPathArrowWidth, \
            "visIsForcePause": self.visIsForcePause, \
            "visForcePauseTime": self.visForcePauseTime, \
            "agentCurrentLoc": [ self.agentCurrentLoc.x, self.agentCurrentLoc.y ], \
            "agentCurrentAct": [ self.agentCurrentAct.dx, self.agentCurrentAct.dy ], \
            "agentLocs": agentLocsList, \
            "agentActs": agentActsList, \
            "isTerminated": self.isTerminated, \
            "nSteps": self.nSteps, \
            "totalValue": self.totalValue
            }

        # Open the file.
        fp = open( strFn, "w" )

        # Save the file.
        json.dump( d, fp, indent=3, sort_keys=True )

        fp.close()

    def load(self, workingDir, fn = None):
        """
        Load the environment from a file.

        if fn == None, a file with the name of GridMapEnv.json will be
        loaded.

        fn is used as locating inside the workding directory.
        """

        if ( not os.path.isdir(workingDir) ):
            raise GridMapException("Working directory {} does not exist.".format(workingDir))

        # Open the file.
        if ( fn is None ):
            strFn = "%s/%s" % ( workingDir, "GridMapEnv.json" )
        else:
            strFn = "%s/%s" % ( workingDir, fn )

        fp = open( strFn, "r" )

        d = json.load( fp )

        fp.close()

        # Update current environment.
        self.workingDir = workingDir
        
        self.name = d["name"]
        self.renderDir = "%s/%s" % ( self.workingDir, "Render" )
        self.maxSteps = d["maxSteps"]
        self.nondimensionalStep = d["nondimensionalStep"]
        self.nondimensionalStepRatio = d["nondimensionalStepRatio"]
        self.flagActionClip = d["flagActionClip"]
        self.actionClip = d["actionClip"]
        self.flagActionValue = d["flagActionValue"]
        self.actionValueFactor = d["actionValueFactor"]
        self.endPointMode = d["endPointMode"]
        self.endPointRadius = d["endPointRadius"]
        self.actStepSize = d["actStepSize"]
        self.normalizedCoordinate = d["normalizedCoordinate"]
        self.isRandomCoordinating = d["isRandomCoordinating"]
        self.randomCoordinatingVariance = d["randomCoordinatingVariance"]

        # Create a new map.
        m = GridMap2D( rows = 1, cols = 1 ) # A temporay map.
        m.read_JSON( self.workingDir + "/" + d["mapFn"] )

        # Set map.
        self.map = m

        # Reset.
        self.reset()

        # Update other member variables.
        self.agentCurrentLoc = BlockCoor( \
            d["agentCurrentLoc"][0], d["agentCurrentLoc"][1] )
        self.agentCurrentAct = BlockCoorDelta( \
            d["agentCurrentAct"][0], d["agentCurrentAct"][1] )
        
        # Agent location history.
        self.agentLocs = []
        for loc in d["agentLocs"]:
            self.agentLocs.append( \
                BlockCoor( loc[0], loc[1] ) )
        
        # Agent action history.
        self.agentActs = []
        for act in d["agentActs"]:
            self.agentActs.append( \
                BlockCoorDelta( act[0], act[1] ) )
        
        # Other member variables.
        self.isTerminated = d["isTerminated"]
        self.nSteps = d["nSteps"]
        self.totalValue = d["totalValue"]
        self.visIsForcePause = d["visIsForcePause"]
        self.visForcePauseTime = d["visForcePauseTime"]

    def can_move_east(self, coor):
        """
        coor is an object of BlockCoor.
        """

        if ( True == self.map.is_east_boundary(coor) ):
            return False
        
        if ( True == self.map.is_north_boundary(coor) or \
             True == self.map.is_south_boundary(coor) ):
            return False
            
        loc = self.map.is_corner_or_principle_line(coor)

        if ( (True == loc[0]) or (True == loc[1]) ):
            if ( isinstance( self.map.get_block( loc[3] ), ObstacleBlock ) ):
                return False
            
            index = copy.deepcopy(loc[3])
            index.r -= 1

            if ( isinstance( self.map.get_block( index ), ObstacleBlock ) ):
                return False
            
            return True
        
        if ( True == loc[2] ):
            if ( isinstance( self.map.get_block( loc[3] ), ObstacleBlock ) ):
                return False

        return True
                
    def can_move_northeast(self, coor):
        """
        coor is an object of BlockCoor.
        """
        
        if ( True == self.map.is_east_boundary(coor) or \
             True == self.map.is_north_boundary(coor) ):
            return False
        
        loc = self.map.is_corner_or_principle_line(coor)

        if ( True == loc[0] ):
            if ( isinstance( self.map.get_block(loc[3]), ObstacleBlock ) ):
                return False
            else:
                return True    
        
        if ( True == loc[1] ):
            if ( isinstance( self.map.get_block(loc[3]), ObstacleBlock ) ):
                return False
            else:
                return True

        if ( True == loc[2] ):
            if ( isinstance( self.map.get_block(loc[3]), ObstacleBlock ) ):
                return False
            else:
                return True

        return True

    def can_move_north(self, coor):
        """
        coor is an object of BlockCoor.
        """

        if ( True == self.map.is_north_boundary(coor) ):
            return False
        
        if ( True == self.map.is_east_boundary(coor) or \
             True == self.map.is_west_boundary(coor) ):
            return False
            
        loc = self.map.is_corner_or_principle_line(coor)

        if ( (True == loc[0]) or (True == loc[2]) ):
            if ( isinstance( self.map.get_block( loc[3] ), ObstacleBlock ) ):
                return False
            
            index = copy.deepcopy(loc[3])
            index.c -= 1

            if ( isinstance( self.map.get_block( index ), ObstacleBlock ) ):
                return False
            
            return True
        
        if ( True == loc[1] ):
            if ( isinstance( self.map.get_block( loc[3] ), ObstacleBlock ) ):
                return False

        return True

    def can_move_northwest(self, coor):
        """
        coor is an object of BlockCoor.
        """
        
        if ( True == self.map.is_west_boundary(coor) or \
             True == self.map.is_north_boundary(coor) ):
            return False
        
        loc = self.map.is_corner_or_principle_line(coor)

        if ( True == loc[0] ):
            index = copy.deepcopy(loc[3])
            index.c -= 1 # Left block.
            if ( isinstance( self.map.get_block(index), ObstacleBlock ) ):
                return False
            else:
                return True    
        
        if ( True == loc[1] ):
            index = copy.deepcopy(loc[3])
            if ( isinstance( self.map.get_block(index), ObstacleBlock ) ):
                return False
            else:
                return True

        if ( True == loc[2] ):
            index = copy.deepcopy(loc[3])
            index.c -= 1 # Left block.
            if ( isinstance( self.map.get_block(index), ObstacleBlock ) ):
                return False
            else:
                return True

        return True

    def can_move_west(self, coor):
        """
        coor is an object of BlockCoor.
        """

        if ( True == self.map.is_west_boundary(coor) ):
            return False
        
        if ( True == self.map.is_north_boundary(coor) or \
             True == self.map.is_south_boundary(coor) ):
            return False
            
        loc = self.map.is_corner_or_principle_line(coor)

        if ( True == loc[0] ):
            index = copy.deepcopy(loc[3])
            index.c -= 1 # Left block.
            
            if ( isinstance( self.map.get_block( index ), ObstacleBlock ) ):
                return False
            
            index.r -= 1 # Now bottom left block.

            if ( isinstance( self.map.get_block( index ), ObstacleBlock ) ):
                return False
            
            return True
        
        if ( True == loc[1] ):
            index = copy.deepcopy(loc[3])
            if ( isinstance( self.map.get_block( index ), ObstacleBlock ) ):
                return False

            index.r -= 1 # Bottom block.
            
            if ( isinstance( self.map.get_block( index ), ObstacleBlock ) ):
                return False

        if ( True == loc[2] ):
            index = copy.deepcopy( loc[3] )
            index.c -= 1 # Left block.

            if ( isinstance( self.map.get_block(index), ObstacleBlock ) ):
                return False

        return True
    
    def can_move_southwest(self, coor):
        """
        coor is an object of BlockCoor.
        """
        
        if ( True == self.map.is_west_boundary(coor) or \
             True == self.map.is_south_boundary(coor) ):
            return False
        
        loc = self.map.is_corner_or_principle_line(coor)

        if ( True == loc[0] ):
            index = copy.deepcopy(loc[3])
            index.c -= 1 # Left block.
            index.r -= 1 # Bottom left block.
            if ( isinstance( self.map.get_block(index), ObstacleBlock ) ):
                return False
            else:
                return True    
        
        if ( True == loc[1] ):
            index = copy.deepcopy(loc[3])
            index.r -= 1 # Bottom block.
            if ( isinstance( self.map.get_block(index), ObstacleBlock ) ):
                return False
            else:
                return True

        if ( True == loc[2] ):
            index = copy.deepcopy(loc[3])
            index.c -= 1 # Left block.
            if ( isinstance( self.map.get_block(index), ObstacleBlock ) ):
                return False
            else:
                return True

        return True
    
    def can_move_south(self, coor):
        """
        coor is an object of BlockCoor.
        """

        if ( True == self.map.is_south_boundary(coor) ):
            return False
        
        if ( True == self.map.is_east_boundary(coor) or \
             True == self.map.is_west_boundary(coor) ):
            return False
            
        loc = self.map.is_corner_or_principle_line(coor)

        if ( True == loc[0] ):
            index = copy.deepcopy(loc[3])
            index.r -= 1 # Bottom block.
            
            if ( isinstance( self.map.get_block( index ), ObstacleBlock ) ):
                return False
            
            index.c -= 1 # Now bottom left block.

            if ( isinstance( self.map.get_block( index ), ObstacleBlock ) ):
                return False
            
            return True
        
        if ( True == loc[2] ):
            index = copy.deepcopy(loc[3])
            if ( isinstance( self.map.get_block( index ), ObstacleBlock ) ):
                return False

            index.c -= 1 # Left block.
            
            if ( isinstance( self.map.get_block( index ), ObstacleBlock ) ):
                return False

        if ( True == loc[1] ):
            index = copy.deepcopy( loc[3] )
            index.r -= 1 # Bottom block.

            if ( isinstance( self.map.get_block(index), ObstacleBlock ) ):
                return False

        return True

    def can_move_southeast(self, coor):
        """
        coor is an object of BlockCoor.
        """
        
        if ( True == self.map.is_east_boundary(coor) or \
             True == self.map.is_south_boundary(coor) ):
            return False
        
        loc = self.map.is_corner_or_principle_line(coor)

        if ( True == loc[0] ):
            index = copy.deepcopy(loc[3])
            index.r -= 1 # Bottom block.
            if ( isinstance( self.map.get_block(index), ObstacleBlock ) ):
                return False
            else:
                return True    
        
        if ( True == loc[1] ):
            index = copy.deepcopy(loc[3])
            index.r -= 1 # Bottom block.
            if ( isinstance( self.map.get_block(index), ObstacleBlock ) ):
                return False
            else:
                return True

        if ( True == loc[2] ):
            if ( isinstance( self.map.get_block(loc[3]), ObstacleBlock ) ):
                return False
            else:
                return True

        return True

    def can_move(self, x, y, dx, dy):
        """Return True if the agent makes a valid line path 
        starts from (x, y) and goes to (x + dx, y + dy). Return False if 
        the agent could not go that direction."""

        coor = BlockCoor(x, y)

        # 8-way switch!
        if ( dx > 0 and dy == 0 ):
            # East direction.
            return self.can_move_east(coor)
        elif ( dx >0 and dy > 0 ):
            # Northeast direction.
            return self.can_move_northeast(coor)
        elif ( dx == 0 and dy > 0 ):
            # North direction.
            return self.can_move_north(coor)
        elif ( dx < 0 and dy > 0 ):
            # Northwest direction.
            return self.can_move_northwest(coor)
        elif ( dx < 0 and dy == 0 ):
            # West direction.
            return self.can_move_west(coor)
        elif ( dx < 0 and dy < 0 ):
            # Southwest direction.
            return self.can_move_southwest(coor)
        elif ( dx == 0 and dy < 0 ):
            # South direction.
            return self.can_move_south(coor)
        elif ( dx > 0 and dy < 0 ):
            # Southeast direction.
            return self.can_move_southeast(coor)
        else:
            print(self.agentCurrentLoc, self.agentCurrentAct)
            return False
            # raise ValueError("dx and dy may not both be zero at the same time.")

    def try_move(self, coorOri, coorDelta):
        """
        coorOri is an object of BlockCoor. Will be deepcopied.
        coorDelta is the delta.

        Return new location coordinate, block value, flag of termination.
        """

        coor = copy.deepcopy(coorOri)
        val  = 0 # The block value.

        # Regularize input coor.
        coor.x = round_if_needed(coor.x)
        coor.y = round_if_needed(coor.y)

        # dx and dy.
        delta = coorDelta.convert_to_direction_delta()

        # Temporary indices.
        idxH = BlockIndex(0, 0)
        idxV = BlockIndex(0, 0)
        coorH = BlockCoor(0, 0)
        coorV = BlockCoor(0, 0)

        # Status monitor.
        tryCount     = 0
        tryCoor      = []
        tryCoorDelta = []

        # Try to move.
        if ( True == self.can_move( coor.x, coor.y, delta.dx, delta.dy ) ):
            coorPre = copy.deepcopy( coor )
            while ( True ):
                # Status monitor.
                tryCoor.append( copy.deepcopy(coor) )
                tryCoorDelta.append( copy.deepcopy(delta) )

                if ( self.tryMoveMaxCount > 0 and tryCount >= self.tryMoveMaxCount ):
                    print("coorOri = %s" % (coorOri) )
                    print("coorDelta = %s" % (coorDelta) )

                    n = len(tryCoor)

                    for i in range(n):
                        print("%s, %s" % (tryCoor[i], tryCoorDelta[i]))
                    
                    raise GridMapException("try_move() reaches its maximum allowed moves.")
                else:
                    tryCount += 1
                
                # Done with status monitor.

                # Get the index of coor.
                index = self.map.get_index_by_coordinates( coor )

                # Get information on coor.
                loc = self.map.is_corner_or_principle_line( coor )

                # Get the targeting vertical and horizontal line index.
                if ( delta.dx >= 0 ):
                    idxV.c = index.c + int( delta.dx )
                else:
                    if ( True == loc[2] ):
                        # Starting from a vertical line.
                        idxV.c = index.c + int( delta.dx )
                    else:
                        idxV.c = index.c

                if ( delta.dy >= 0 ):
                    idxH.r = index.r + int( delta.dy )
                else:
                    if ( True == loc[1] ):
                        # Starting from a horizontal line.
                        idxH.r = index.r + int( delta.dy )
                    else:
                        idxH.r = index.r

                # Get the x coordinates for the vertical line.
                coorV = self.map.convert_to_coordinates( idxV )
                # Get the y coordinates for the horizontal line.
                coorH = self.map.convert_to_coordinates( idxH )

                # Find two possible intersections with these lines.
                [xV, yV], flagV = LineIntersection2D.line_intersect( \
                    coorOri.x, coorOri.y, coorOri.x + coorDelta.dx, coorOri.y + coorDelta.dy, \
                    coorV.x, self.map.corners[0][GridMap2D.I_Y], coorV.x, self.map.corners[3][GridMap2D.I_Y] )

                xV = round_if_needed(xV)
                yV = round_if_needed(yV)

                [xH, yH], flagH = LineIntersection2D.line_intersect( \
                    coorOri.x, coorOri.y, coorOri.x + coorDelta.dx, coorOri.y + coorDelta.dy, \
                    self.map.corners[0][GridMap2D.I_X], coorH.y, self.map.corners[1][GridMap2D.I_X], coorH.y )
                
                xH = round_if_needed(xH)
                yH = round_if_needed(yH)

                if ( LineIntersection2D.VALID_INTERSECTION == flagV ):
                    distV = two_point_distance( coor.x, coor.y, xV, yV )
                else:
                    distV = 0

                if ( LineIntersection2D.VALID_INTERSECTION == flagH ): 
                    distH = two_point_distance( coor.x, coor.y, xH, yH )
                else:
                    distH = 0
                    
                # Auxiliary check.
                auxV = self.map.is_corner_or_principle_line( BlockCoor( xV, yV ) )
                auxH = self.map.is_corner_or_principle_line( BlockCoor( xH, yH ) )

                if ( LineIntersection2D.VALID_INTERSECTION == flagV and \
                     True == auxV[0] and \
                     LineIntersection2D.VALID_INTERSECTION == flagH and \
                     True == auxH[0] ):
                    # Same distance.
                    # Choose a pair of valid coordinates that are not None.
                    if ( xV is not None and yV is not None ):
                        xi = xV; yi = yV
                    elif ( xH is not None and yH is not None ):
                        xi = xH; yi = yH
                    else:
                        raise GridMapException( "Vertical and horizontal intersections must not both be invalid." )

                    # Check if (xi, yi) is on the boundary.
                    if ( True == self.map.is_out_of_or_on_boundary( BlockCoor( xi, yi ) ) ):
                        # Stop here.
                        coorPre.x, coorPre.y = coor.x, coor.y
                        coor.x, coor.y = xi, yi
                        break
                    
                    # Get the index at (xi, yi). 
                    interIdxH = self.map.get_index_by_coordinates( BlockCoor(xi, yi) )

                    # Since we are at a corner point, we simply checkout all four neighboring blocks.
                    flagCornerFoundObstacle = False

                    if ( False == flagCornerFoundObstacle and \
                            True == self.map.is_obstacle_block( interIdxH ) ):
                        flagCornerFoundObstacle = True
                    
                    interIdxH.c -= 1
                    if ( False == flagCornerFoundObstacle and \
                            True == self.map.is_obstacle_block( interIdxH ) ):
                        flagCornerFoundObstacle = True
                    
                    interIdxH.r -= 1
                    if ( False == flagCornerFoundObstacle and \
                            True == self.map.is_obstacle_block( interIdxH ) ):
                        flagCornerFoundObstacle = True
                    
                    interIdxH.c += 1
                    if ( False == flagCornerFoundObstacle and \
                            True == self.map.is_obstacle_block( interIdxH ) ):
                        flagCornerFoundObstacle = True
                    
                    if ( True == flagCornerFoundObstacle ):
                        # Stop here.
                        coorPre.x, coorPre.y = coor.x, coor.y
                        coor.x, coor.y = xi, yi
                        break
                    
                    coorPre.x, coorPre.y = coor.x, coor.y
                    coor.x, coor.y = xi, yi
                    continue

                if ( LineIntersection2D.VALID_INTERSECTION == flagV ):
                    if ( LineIntersection2D.VALID_INTERSECTION != flagH or \
                         distV < distH ):
                        # Check if (xV, yV) is on the boundary.
                        if ( True == self.map.is_out_of_or_on_boundary( BlockCoor( xV, yV ) ) ):
                            # Stop here.
                            coorPre.x, coorPre.y = coor.x, coor.y
                            coor.x, coor.y = xV, yV
                            break
                        
                        # Get the index at (xV, yV).
                        interIdxV = self.map.get_index_by_coordinates( BlockCoor(xV, yV) )

                        if ( delta.dx < 0 ):
                            # Left direction.
                            interIdxV.c -= 1

                        if ( True == self.map.is_obstacle_block( interIdxV ) ):
                            # Stop here.
                            coorPre.x, coorPre.y = coor.x, coor.y
                            coor.x, coor.y = xV, yV
                            break

                        # Check if we are travelling along a horizontal line.
                        if ( loc[1] == True and delta.dy == 0 ):
                            # South direction.
                            interIdxV.r -= 1

                        if ( True == self.map.is_obstacle_block( interIdxV ) ):
                            # Stop here.
                            coorPre.x, coorPre.y = coor.x, coor.y
                            coor.x, coor.y = xV, yV
                            break
                        
                        coorPre.x, coorPre.y = coor.x, coor.y
                        coor.x, coor.y = xV, yV
                        continue
                        
                if ( LineIntersection2D.VALID_INTERSECTION == flagH ):
                    # Not same distance.
                    # Check if (xH, yH) is on the boundary.
                    if ( True == self.map.is_out_of_or_on_boundary( BlockCoor( xH, yH ) ) ):
                        # Stop here.
                        coorPre.x, coorPre.y = coor.x, coor.y
                        coor.x, coor.y = xH, yH
                        break
                    
                    # Get the index at (xH, yH).
                    interIdxH = self.map.get_index_by_coordinates( BlockCoor(xH, yH) )

                    if ( delta.dy < 0 ):
                        # Downwards direction.
                        interIdxH.r -= 1

                    if ( True == self.map.is_obstacle_block( interIdxH ) ):
                        # Stop here.
                        coorPre.x, coorPre.y = coor.x, coor.y
                        coor.x, coor.y = xH, yH
                        break
                    
                    # Check if we are travelling along a vertical line.
                    if ( loc[2] == True and delta.dx == 0 ):
                        interIdxH.c -= 1
                    
                    if ( True == self.map.is_obstacle_block( interIdxH ) ):
                        # Stop here.
                        coorPre.x, coorPre.y = coor.x, coor.y
                        coor.x, coor.y = xH, yH
                        break
                    
                    coorPre.x, coorPre.y = coor.x, coor.y
                    coor.x, coor.y = xH, yH
                    continue
                
                # No valid intersectons. Stop here.
                coorPre.x, coorPre.y = coor.x, coor.y

                coor.x = coorOri.x + coorDelta.dx
                coor.y = coorOri.y + coorDelta.dy

                # coor.x = round_if_needed(coor.x)
                # coor.y = round_if_needed(coor.y)

                break

            # Check if coor is out of boundary.
            if ( True == self.map.is_out_of_boundary( coor ) ):
                s = "Out of boundary before evaluation. coorOri = %s, coorDelta - %s" % \
                    (coorOri, coorDelta)
                print(s)
                # raise GridMapException( s )

                # Roll back to the previous valid intersection.
                coor.x, coor.y = coorPre.x, coorPre.y
                print( "Rolled back to previous coordinate (%f, %f)" % ( coor.x, coor.y ) )

            if ( False == self.map.is_in_ending_block(coor) ):
                val = self.map.evaluate_coordinate( coor )
        else:
            # Cannot move.
            val = self.map.evaluate_coordinate( coor )

        # Check if it is in the ending block.
        flagTerm = False

        if ( GridMapEnv.END_POINT_MODE_BLOCK == self.endPointMode ):
            if ( True == self.map.is_in_ending_block( coor ) ):
                flagTerm = True
                val += self.map.valueEndingBlock
        elif ( GridMapEnv.END_POINT_MODE_RADIUS == self.endPointMode ):
            if ( True == self.map.is_around_ending_block( coor, self.endPointRadius ) ):
                flagTerm = True
                val += self.map.valueEndingBlock
        else:
            raise GridMapException("Unexpected self.endPointMode. self.endPointMode = {}".format(self.endPointMode))

        return coor, val, flagTerm

    def get_string_agent_locs(self):
        s = ""

        for loc in self.agentLocs:
            s += "(%f, %f)\n" % (loc.x, loc.y)

        return s

    def get_string_agent_acts(self):
        s = ""

        for act in self.agentActs:
            s += "(%f, %f)\n" % (act.dx, act.dy)
        
        return s

    def __str__(self):
        s = "GridMapEnv %s\n" % (self.name)

        # Map.
        if ( self.map is None ):
            s += "(No map)\n"
        else:
            s += "Map: %s\n" % (self.map.name)
        
        s += "Working directory: %s\nRender directory: %s\n" % (self.workingDir, self.renderDir)
        s += "maxSteps = %d\n" % (self.maxSteps)

        s += "visAgentRadius = %f\n" % (self.visAgentRadius)
        s += "visPathArrowWidth = %f\n" % (self.visPathArrowWidth)

        s += "nSteps = %d\n" % (self.nSteps)
        s += "totalValue = %f\n" % ( self.totalValue )
        s += "isTerminated = {}\n".format( self.isTerminated )

        s += "agentCurrentLoc: {}\n".format( self.agentCurrentLoc )
        s += "agentCurrentAct: {}\n".format( self.agentCurrentAct )

        # History of locations and actions.
        s += "agentLocs: \n%s\n" % ( self.get_string_agent_locs() )
        s += "agentActs: \n%s\n" % ( self.get_string_agent_acts() )

        return s

if __name__ == "__main__":
    print("Hello GridMap.")

    # Create a GridMap2D object.
    gm2d = GridMap2D(10, 20, outOfBoundValue=-200)
    gm2d.initialize()

    # Create a starting block and an ending block.
    startingBlock = StartingBlock()
    endingBlock   = EndingBlock()

    # Create an obstacle block.
    obstacle = ObstacleBlock()

    # Overwrite blocks.
    gm2d.set_starting_block((0, 0))
    gm2d.set_ending_block((9, 19), endPoint=(19.1, 9.1))
    gm2d.add_obstacle((4, 10))
    gm2d.add_obstacle((5, 10))
    gm2d.add_obstacle((6, 10))

    indexEndingBlock = gm2d.get_index_ending_block()
    ebGm2d = gm2d.get_block(indexEndingBlock)

    print("ebGm2d.is_in_range(19.2, 9.2, 1) = {}".format( ebGm2d.is_in_range(19.2, 9.2, 1) ) )

    # Describe the map.
    print(gm2d)

    # import ipdb; ipdb.set_trace()

    # Test GridMap2D.evaluate_coordinate
    print("Value of (   0,      0) is %f" % ( gm2d.evaluate_coordinate( (0, 0) ) ) )
    print("Value of (19.99,  9.99) is %f" % ( gm2d.evaluate_coordinate( (19.99, 9.99) ) ) )
    print("Value of (19.99,     0) is %f" % ( gm2d.evaluate_coordinate( (19.99, 0) ) ) )
    print("Value of (    0,  9.99) is %f" % ( gm2d.evaluate_coordinate( (0, 9.99) ) ) )
    print("Value of (   10,     4) is %f" % ( gm2d.evaluate_coordinate( (10, 4) ) ) )
    print("Value of (   10,     5) is %f" % ( gm2d.evaluate_coordinate( (10, 5) ) ) )
    print("Value of (   10,     6) is %f" % ( gm2d.evaluate_coordinate( (10, 6) ) ) )
    print("Value of (   10,   5.5) is %f" % ( gm2d.evaluate_coordinate( (10, 5.5) ) ) )
    print("Value of ( 10.5,     5) is %f" % ( gm2d.evaluate_coordinate( (10.5, 5) ) ) )
    print("Value of (10.99,  5.99) is %f" % ( gm2d.evaluate_coordinate( (10.99, 5.99) ) ) )
    # print("Value of (   -1,    -1) is %f" % ( gm2d.evaluate_coordinate( (-1, -1) ) ) )
    # print("Value of (    9, -0.01) is %f" % ( gm2d.evaluate_coordinate( (9, -0.01) ) ) )
    # print("Value of (    9, 10.01) is %f" % ( gm2d.evaluate_coordinate( (9, 10.01) ) ) )
    # print("Value of (-0.01,     5) is %f" % ( gm2d.evaluate_coordinate( (-0.01, 5) ) ) )
    # print("Value of (20.01,     5) is %f" % ( gm2d.evaluate_coordinate( (20.01, 5) ) ) )

    # Create a GridMapEnv object.
    gme = GridMapEnv(gridMap = gm2d)

    # Render.
    gme.render()
