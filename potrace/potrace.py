# =============================================================================
# Potrace Python Implementation
# =============================================================================
# This is a Python implementation of the Potrace algorithm for converting
# bitmap images into vector graphics. The algorithm traces the boundaries
# of black regions in a bitmap and converts them into smooth vector curves.
# =============================================================================

import math
from typing import Optional, Tuple, Union

import numpy as np

# =============================================================================
# TURN POLICY CONSTANTS
# =============================================================================
# These constants define different strategies for resolving ambiguous cases
# during path tracing when multiple directions are equally valid.

POTRACE_TURNPOLICY_BLACK = 0    # Prefer black pixels (keep black to the left)
POTRACE_TURNPOLICY_WHITE = 1    # Prefer white pixels (keep white to the left)
POTRACE_TURNPOLICY_LEFT = 2     # Always turn left when ambiguous
POTRACE_TURNPOLICY_RIGHT = 3    # Always turn right when ambiguous
POTRACE_TURNPOLICY_MINORITY = 4 # Choose the direction that occurs less often
POTRACE_TURNPOLICY_MAJORITY = 5 # Choose the direction that occurs more often
POTRACE_TURNPOLICY_RANDOM = 6   # Use deterministic random choice

# =============================================================================
# SEGMENT TYPE CONSTANTS
# =============================================================================
# These define the types of curve segments that can be created

POTRACE_CURVETO = 1  # Smooth curve segment (Bezier curve)
POTRACE_CORNER = 2   # Sharp corner segment (straight line)

# =============================================================================
# MATHEMATICAL CONSTANTS
# =============================================================================

INFTY = float("inf")  # Infinity value for mathematical calculations
COS179 = math.cos(math.radians(179))  # Cosine of 179 degrees (used for angle checks)


# =============================================================================
# BITMAP CLASS
# =============================================================================
# Main class for processing bitmap images and converting them to vector paths.
# Handles input conversion, bitmap manipulation, and the main tracing process.

class Bitmap:
    """
    Represents a bitmap image and provides methods to trace it into vector paths.
    
    The Bitmap class handles:
    - Input format conversion (numpy arrays, PIL images, etc.)
    - Bitmap preprocessing and inversion
    - Main tracing process with configurable parameters
    """
    
    def __init__(self, data, blacklevel=0.5):
        """
        Initialize a Bitmap from various input formats.
        
        Args:
            data: Input data in various formats:
                  - numpy array (2D boolean or numeric)
                  - PIL Image object
                  - Any array-like object
            blacklevel: Threshold for converting grayscale to binary (default: 0.5)
                       Pixels darker than this threshold become black
        """
        # Handle numpy arrays with different data types
        if hasattr(data, "dtype"):
            if data.dtype != "bool":
                # Convert numeric arrays to boolean using threshold
                data = data > (255 * blacklevel)
        
        # Handle PIL Image objects
        if hasattr(data, "mode"):
            if data.mode != "L":
                # Convert to grayscale if not already
                data = data.convert("L")
            # Apply threshold to convert to binary
            data = data.point(lambda e: 0 if (e / 255.0) < blacklevel else 255)
            # Convert to 1-bit image and then to numpy array
            image = data.convert("1")
            data = np.array(image)
        
        # Store the processed bitmap data
        self.data = data
        # Invert the bitmap (Potrace works with white=1, black=0)
        self.invert()

    def invert(self):
        """
        Invert the bitmap (swap black and white).
        Potrace algorithm expects white pixels to be 1 and black pixels to be 0.
        """
        self.data = np.invert(self.data)

    def trace(
        self,
        turdsize: int = 2,
        turnpolicy: int = POTRACE_TURNPOLICY_MINORITY,
        alphamax=1.0,
        opticurve=True,
        opttolerance=0.2,
    ):
        """
        Main method to trace the bitmap and convert it to vector paths.
        
        Args:
            turdsize: Minimum area for paths (smaller paths are ignored)
            turnpolicy: Strategy for resolving ambiguous turns during path tracing
            alphamax: Maximum smoothness parameter (0.0 to 1.0)
            opticurve: Whether to optimize curves by combining segments
            opttolerance: Tolerance for curve optimization
            
        Returns:
            Path object containing the traced vector curves
        """
        # Pad the bitmap with one row/column of zeros for boundary handling
        bm = np.pad(self.data, [(0, 1), (0, 1)], mode="constant")

        # Stage 1: Decompose bitmap into individual paths
        plist = bm_to_pathlist(bm, turdsize=turdsize, turnpolicy=turnpolicy)
        
        # Stages 2-5: Process paths into smooth curves
        process_path(
            plist,
            alphamax=alphamax,
            opticurve=opticurve,
            opttolerance=opttolerance,
        )
        
        # Return the final Path object containing all curves
        return Path(plist)


# =============================================================================
# PATH CLASS
# =============================================================================
# Container class that holds multiple curves forming a complete vector path.
# Inherits from list to provide easy iteration over curves.

class Path(list):
    """
    Container for multiple curves that form a complete vector path.
    Inherits from list to provide easy iteration and list operations.
    """
    
    def __init__(self, plist):
        """
        Initialize Path from a list of processed path objects.
        
        Args:
            plist: List of _Path objects from the processing pipeline
        """
        list.__init__(self)
        # Convert each processed path into a Curve object
        self.extend([Curve(p) for p in plist])

    @property
    def curves(self):
        """
        Property to access the curves in this path.
        Returns the list of curves (same as self).
        """
        return self

    @property
    def curves_tree(self):
        """
        Property for hierarchical curve structure (not implemented).
        Would provide parent-child relationships for nested paths.
        """
        return None


# =============================================================================
# CURVE CLASS
# =============================================================================
# Represents a single closed curve with multiple segments.
# Each curve can contain both straight line segments and Bezier curve segments.

class Curve(list):
    """
    Represents a single closed curve with multiple segments.
    Inherits from list to provide easy iteration over segments.
    """
    
    def __init__(self, p):
        """
        Initialize Curve from a processed path object.
        
        Args:
            p: _Path object containing the processed curve data
        """
        list.__init__(self)
        last = None
        self._path = p
        self._curve = p._fcurve  # Final curve data
        
        # Convert each segment in the curve to appropriate segment type
        for s in self._curve:
            if s.tag == POTRACE_CORNER:
                # Sharp corner - create straight line segment
                self.append(CornerSegment(s))
            else:
                # Smooth curve - create Bezier curve segment
                self.append(BezierSegment(s))
            last = s
        
        # Store the starting point of the curve
        self.start_point = last.c[2] if last is not None else None

    @property
    def decomposition_points(self):
        """
        Property to access the original path points from bitmap decomposition.
        These are the raw points extracted from the bitmap boundary.
        """
        return self._path.pt

    @property
    def segments(self):
        """
        Property to access the segments in this curve.
        Returns the list of segments (same as self).
        """
        return self

    @property
    def children(self):
        """
        Property for child curves (not implemented).
        Would provide hierarchical structure for nested curves.
        """
        return None


# =============================================================================
# SEGMENT CLASSES
# =============================================================================
# These classes represent individual segments within a curve.
# CornerSegment represents straight lines, BezierSegment represents curved lines.

class CornerSegment:
    """
    Represents a straight line segment (sharp corner) in a curve.
    """
    
    def __init__(self, s):
        """
        Initialize from a segment object.
        
        Args:
            s: _Segment object containing the segment data
        """
        self._segment = s

    @property
    def c(self):
        """
        Property to access the corner point.
        Returns the vertex point of the corner.
        """
        return self._segment.c[1]

    @property
    def end_point(self):
        """
        Property to access the end point of this segment.
        """
        return self._segment.c[2]

    @property
    def is_corner(self):
        """
        Property indicating this is a corner segment.
        Always returns True for CornerSegment.
        """
        return True


class BezierSegment:
    """
    Represents a Bezier curve segment (smooth curve) in a curve.
    """
    
    def __init__(self, s):
        """
        Initialize from a segment object.
        
        Args:
            s: _Segment object containing the segment data
        """
        self._segment = s

    @property
    def c1(self):
        """
        Property to access the first control point of the Bezier curve.
        """
        return self._segment.c[0]

    @property
    def c2(self):
        """
        Property to access the second control point of the Bezier curve.
        """
        return self._segment.c[1]

    @property
    def end_point(self):
        """
        Property to access the end point of this Bezier curve segment.
        """
        return self._segment.c[2]

    @property
    def is_corner(self):
        """
        Property indicating this is not a corner segment.
        Always returns False for BezierSegment.
        """
        return False


# =============================================================================
# INTERNAL DATA STRUCTURES
# =============================================================================
# These classes are used internally during the processing pipeline.
# They are not part of the public API.

class _Curve:
    """
    Internal curve representation used during processing.
    Contains segments and processing state.
    """
    
    def __init__(self, m):
        """
        Initialize with a specified number of segments.
        
        Args:
            m: Number of segments in the curve
        """
        self.segments = [_Segment() for _ in range(m)]
        self.alphacurve = False  # Whether curve has been smoothed

    def __len__(self):
        """Return the number of segments in this curve."""
        return len(self.segments)

    @property
    def n(self):
        """Property to get the number of segments."""
        return len(self)

    def __getitem__(self, item):
        """Allow indexing to access segments."""
        return self.segments[item]


class _Path:
    """
    Internal path representation used during processing.
    Contains all the data needed for the multi-stage processing pipeline.
    """
    
    def __init__(self, pt: list, area: int, sign: bool):
        """
        Initialize a path with its basic properties.
        
        Args:
            pt: List of points defining the path boundary
            area: Area enclosed by the path (positive for black regions)
            sign: Whether path is positive (True) or negative (False)
        """
        self.pt = pt  # List of points defining the path boundary
        
        # Basic path properties
        self.area = area      # Area enclosed by the path
        self.sign = sign      # True for positive paths (black regions)
        
        # Tree structure properties (for nested paths - not fully implemented)
        self.next = None      # Next path in linked list
        self.childlist = []   # Child paths (nested inside this path)
        self.sibling = []     # Sibling paths (at same level)

        # Processing data for Stage 1: Longest line calculation
        self._lon = []        # Longest straight line from each point

        # Processing data for Stage 2: Polygon calculation
        self._x0 = 0          # Origin point for coordinate calculations
        self._y0 = 0          # Origin point for coordinate calculations
        self._sums = []       # Precomputed sums for fast calculations

        # Processing data for Stage 3-5: Curve generation
        self._m = 0           # Number of vertices in optimal polygon
        self._po = []         # Optimal polygon vertices
        self._curve = []      # Initial curve segments
        self._ocurve = []     # Optimized curve segments
        self._fcurve = []     # Final curve (points to either _curve or _ocurve)

    def __len__(self):
        """Return the number of points in this path."""
        return len(self.pt)

    def init_curve(self, m):
        """
        Initialize curve data (placeholder method).
        
        Args:
            m: Number of segments to initialize
        """
        pass


class _Point:
    """
    Simple 2D point representation used throughout the algorithm.
    """
    
    def __init__(self, x: float = 0, y: float = 0):
        """
        Initialize a point with x,y coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
        """
        self.x = x
        self.y = y

    def __repr__(self):
        """String representation of the point."""
        return "Point(%f, %f)" % (self.x, self.y)


class _Segment:
    """
    Internal segment representation used during curve processing.
    Contains all the data needed for both corner and Bezier segments.
    """
    
    def __init__(self):
        """
        Initialize a segment with default values.
        The segment type and control points are set during processing.
        """
        self.tag = 0          # Type: POTRACE_CORNER or POTRACE_CURVETO
        self.c = [_Point(), _Point(), _Point()]  # Control points for Bezier curves
        self.vertex = _Point()  # Vertex point of the segment
        self.alpha = 0.0      # Smoothness parameter (0.0 to 1.0)
        self.alpha0 = 0.0     # Original alpha value before optimization
        self.beta = 0.0       # Curve parameter for optimization


class _Sums:
    """
    Container for precomputed sums used in fast calculations.
    These sums are used in Stage 2 for polygon calculations.
    """
    
    def __init__(self):
        """
        Initialize all sum values to zero.
        These will be filled during the _calc_sums() function.
        """
        self.x = 0    # Sum of x coordinates
        self.y = 0    # Sum of y coordinates
        self.x2 = 0   # Sum of x² coordinates
        self.xy = 0   # Sum of x*y coordinates
        self.y2 = 0   # Sum of y² coordinates


# =============================================================================
# DETERMINISTIC RANDOM NUMBER GENERATION
# =============================================================================
# This lookup table provides deterministic pseudo-random values for resolving
# ambiguous cases during path tracing. It's based on Galois Field arithmetic.

detrand_t = (
    # /* non-linear sequence: constant term of inverse in GF(8),
    #   mod x^8+x^4+x^3+x+1 */
    # This is a 256-element lookup table for deterministic random number generation.
    # Each value is the constant term of the inverse in GF(8) with polynomial x^8+x^4+x^3+x+1.
    # Used by the detrand() function to generate consistent "random" values.
    0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1,
    0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0,
    0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0,
    0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1,
    1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1,
    1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1,
    0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1,
    1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0,
    0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1,
    1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1,
    1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
)


# =============================================================================
# STAGE 1: PATH DECOMPOSITION FUNCTIONS
# =============================================================================
# These functions implement Stage 1 of the Potrace algorithm:
# decomposing the bitmap into individual paths by tracing boundaries.

def detrand(x: int, y: int) -> int:
    """
    Deterministically and efficiently hash (x,y) into a pseudo-random bit.
    
    This function provides deterministic "random" values for resolving ambiguous
    cases during path tracing. It uses the detrand_t lookup table to generate
    consistent results for the same input coordinates.
    
    Args:
        x: X coordinate
        y: Y coordinate
        
    Returns:
        A pseudo-random bit (0 or 1) that is deterministic for the same input
    """
    # /* 0x04b3e375 and 0x05a8ef93 are chosen to contain every possible 5-bit sequence */
    # These magic numbers ensure good distribution of the hash function
    z = ((0x04B3E375 * x) ^ y) * 0x05A8EF93
    
    # Use the lookup table to convert each byte of the hash into a random bit
    # XOR the results to produce the final random bit
    z = (
        detrand_t[z & 0xFF]           # Use lowest byte
        ^ detrand_t[(z >> 8) & 0xFF]  # Use second byte
        ^ detrand_t[(z >> 16) & 0xFF] # Use third byte
        ^ detrand_t[(z >> 24) & 0xFF] # Use highest byte
    )
    return z


def majority(bm: np.array, x: int, y: int) -> int:
    """
    Return the "majority" value of bitmap bm at intersection (x,y).
    
    This function examines the neighborhood around a point to determine
    whether it should be considered black or white when there's ambiguity.
    It assumes that the bitmap is balanced at "radius" 1.
    
    Args:
        bm: The bitmap array
        x: X coordinate of the point
        y: Y coordinate of the point
        
    Returns:
        1 if the majority of the neighborhood is black, 0 if white
    """
    # Check at increasing "radius" until we find a clear majority
    for i in range(2, 5):  # Check at radius i
        ct = 0  # Counter for black vs white pixels
        
        # Check the four sides of the square at radius i
        for a in range(-i + 1, i - 2):
            # Check top side
            try:
                ct += 1 if bm[y + i - 1][x + a] else -1
            except IndexError:
                pass  # Ignore out-of-bounds pixels
            # Check right side
            try:
                ct += 1 if bm[y + a - 1][x + i - 1] else -1
            except IndexError:
                pass
            # Check bottom side
            try:
                ct += 1 if bm[y - i][x + a - 1] else -1
            except IndexError:
                pass
            # Check left side
            try:
                ct += 1 if bm[y + a][x - i] else -1
            except IndexError:
                pass
        
        # If we found a clear majority, return it
        if ct > 0:
            return 1  # Majority is black
        elif ct < 0:
            return 0  # Majority is white
    
    # If no clear majority found, default to white
    return 0


def xor_to_ref(bm: np.array, x: int, y: int, xa: int) -> None:
    """
    Efficiently invert bits [x,infty) and [xa,infty) in line y.
    
    This function is used to efficiently flip pixels in a horizontal line
    during path removal. Here xa must be a multiple of BM_WORDBITS.
    
    Args:
        bm: The bitmap array to modify
        x: Start position for first inversion
        y: Y coordinate of the line
        xa: Start position for second inversion
    """
    # Invert the range between x and xa
    if x < xa:
        bm[y, x:xa] ^= True
    elif x != xa:
        bm[y, xa:x] ^= True


def xor_path(bm: np.array, p: _Path) -> None:
    """
    XOR the given pixmap with the interior of the given path.
    
    A path is represented as an array of points, which are thought to
    lie on the corners of pixels (not on their centers). The path point
    (x,y) is the lower left corner of the pixel (x,y). Paths are
    represented by the len/pt components of a path_t object.
    
    This function removes the traced path from the bitmap by XORing
    the interior of the path with the bitmap. This is used to ensure
    that each region is traced only once.
    
    Args:
        bm: The bitmap array to modify
        p: The path object containing the path to remove
        
    Note:
        The path must be within the dimensions of the pixmap.
    """
    if len(p) <= 0:  # A path of length 0 is silly, but legal
        return

    y1 = p.pt[-1].y  # Previous Y coordinate
    xa = p.pt[0].x   # Starting X coordinate
    
    # Process each point in the path
    for n in p.pt:
        x, y = n.x, n.y
        if y != y1:
            # Efficiently invert the rectangle [x,xa] x [y,y1]
            xor_to_ref(bm, x, min(y, y1), xa)
            y1 = y


def findpath(bm: np.array, x0: int, y0: int, sign: bool, turnpolicy: int) -> _Path:
    """
    Compute a path in the given pixmap, separating black from white.
    
    This is the core path tracing algorithm that follows the boundary of a region
    using the "left-hand rule". It starts at an upper left corner and traces
    the complete boundary until returning to the starting point.
    
    Args:
        bm: The bitmap array to trace
        x0: Starting X coordinate (must be an upper left corner)
        y0: Starting Y coordinate (must be an upper left corner)
        sign: Whether this is a positive (True) or negative (False) path
        turnpolicy: Strategy for resolving ambiguous turns
        
    Returns:
        A new _Path object containing the traced path and its properties
        
    Note:
        A legitimate path cannot have length 0. The starting point must be
        an upper left corner of the region to be traced.
    """
    # Initialize path tracing variables
    x = x0
    y = y0
    dirx = 0      # Current X direction
    diry = -1     # Current Y direction (start moving up)
    pt = []       # List to store path points
    area = 0      # Area enclosed by the path

    while True:  # Continue until we return to the starting point
        # Add current point to the path
        pt.append(_Point(int(x), int(y)))

        # Move to the next point along the boundary
        x += dirx
        y += diry
        area += x * diry  # Accumulate area using shoelace formula

        # Check if we've completed the path (returned to start)
        if x == x0 and y == y0:
            break

        # Determine the next direction based on the current position
        # Check the two possible directions: right and left of current direction
        
        # Check right direction (clockwise)
        cy = y + (diry - dirx - 1) // 2
        cx = x + (dirx + diry - 1) // 2
        try:
            c = bm[cy][cx]  # Pixel to the right
        except IndexError:
            c = 0  # Out of bounds = white
            
        # Check left direction (counter-clockwise)
        dy = y + (diry + dirx - 1) // 2
        dx = x + (dirx - diry - 1) // 2
        try:
            d = bm[dy][dx]  # Pixel to the left
        except IndexError:
            d = 0  # Out of bounds = white

        # Determine which direction to turn based on pixel values and turn policy
        if c and not d:  # Ambiguous turn - both directions are valid
            # Apply turn policy to resolve ambiguity
            if (
                turnpolicy == POTRACE_TURNPOLICY_RIGHT
                or (turnpolicy == POTRACE_TURNPOLICY_BLACK and sign)
                or (turnpolicy == POTRACE_TURNPOLICY_WHITE and not sign)
                or (turnpolicy == POTRACE_TURNPOLICY_RANDOM and detrand(x, y))
                or (turnpolicy == POTRACE_TURNPOLICY_MAJORITY and majority(bm, x, y))
                or (
                    turnpolicy == POTRACE_TURNPOLICY_MINORITY and not majority(bm, x, y)
                )
            ):
                # Turn right (clockwise)
                tmp = dirx
                dirx = diry
                diry = -tmp
            else:
                # Turn left (counter-clockwise)
                tmp = dirx
                dirx = -diry
                diry = tmp
        elif c:  # Only right direction is valid - turn right
            tmp = dirx
            dirx = diry
            diry = -tmp
        elif not d:  # Only left direction is valid - turn left
            tmp = dirx
            dirx = -diry
            diry = tmp

    # Create and return the path object
    return _Path(pt, area, sign)


def findnext(bm: np.array) -> Optional[Tuple[Union[int], int]]:
    """
    Find the next set pixel in the bitmap.
    
    Pixels are searched first left-to-right, then top-down. In other words,
    (x,y) < (x',y') if y > y' or y = y' and x < x'. This ensures we find
    the leftmost pixel in the topmost row that contains black pixels.
    
    Args:
        bm: The bitmap array to search
        
    Returns:
        Tuple of (y, x) coordinates of the next black pixel, or None if none found
        
    Note:
        This function assumes that excess bytes have been cleared with bm_clearexcess.
    """
    # Find all non-zero (black) pixels in the bitmap
    w = np.nonzero(bm)
    if len(w[0]) == 0:
        return None  # No black pixels found

    # Find the topmost row that contains black pixels
    q = np.where(w[0] == w[0][-1])
    y = w[0][q]  # Y coordinates of pixels in topmost row
    x = w[1][q]  # X coordinates of pixels in topmost row
    
    # Return the leftmost pixel in the topmost row
    return y[0], x[0]


def setbbox_path(p: _Path):
    """
    Find the bounding box of a given path.
    
    Calculates the minimum and maximum x,y coordinates that contain
    all points in the path. This is used for various optimizations
    and calculations throughout the algorithm.
    
    Args:
        p: The path object to analyze
        
    Returns:
        Tuple of (x0, y0, x1, y1) where (x0,y0) is the top-left corner
        and (x1,y1) is the bottom-right corner of the bounding box
        
    Note:
        Path is assumed to be of non-zero length.
    """
    # Initialize bounding box to extreme values
    y0 = float("inf")  # Minimum Y
    y1 = 0             # Maximum Y
    x0 = float("inf")  # Minimum X
    x1 = 0             # Maximum X
    
    # Scan all points in the path to find min/max coordinates
    for k in range(len(p)):
        x = p.pt[k].x
        y = p.pt[k].y

        # Update bounding box
        if x < x0:
            x0 = x
        if x > x1:
            x1 = x
        if y < y0:
            y0 = y
        if y > y1:
            y1 = y
    
    return x0, y0, x1, y1


def pathlist_to_tree(plist: list, bm: np.array) -> None:
    """
    Give a tree structure to the given path list, based on "insideness" testing.
    
    This function creates a hierarchical structure where path A is considered
    "below" path B if it is inside path B. The input pathlist is assumed to
    be ordered so that "outer" paths occur before "inner" paths.
    
    The tree structure is stored in the "childlist" and "sibling" components
    of the path_t structure. The linked list structure is also changed so that
    negative path components are listed immediately after their positive parent.
    
    Args:
        plist: List of paths to organize into a tree
        bm: Bitmap of the correct size, used as scratch space
        
    Note:
        Some backends may ignore the tree structure, others may use it
        e.g. to group path components. We assume that in the input,
        point 0 of each path is an "upper left" corner of the path,
        as returned by bm_to_pathlist. This makes it easy to find
        an "interior" point.
    """
    # This function is currently not fully implemented in this Python version
    # The original C implementation would:
    # 1. Save original "next" pointers
    # 2. Use a heap-based algorithm to process paths
    # 3. Test each path for insideness relative to others
    # 4. Build parent-child relationships
    # 5. Reconstruct the linked list with proper ordering
    
    # For now, we just save the original pointers and leave the tree structure
    # unimplemented, as indicated by the commented code below
    
    bm = bm.copy()

    # Save original "next" pointers
    for p in plist:
        p.sibling = p.next
        p.childlist = None

    # The heap holds a list of lists of paths. Use "childlist" field
    # for outer list, "next" field for inner list. Each of the sublists
    # is to be turned into a tree. This code is messy, but it is
    # actually fast. Each path is rendered exactly once. We use the
    # heap to get a tail recursive algorithm: the heap holds a list of
    # pathlists which still need to be transformed.
    
    # The following code is commented out because the full tree structure
    # implementation is complex and not essential for basic path tracing
    
    """
    heap = plist
    while heap:
        # Unlink first sublist
        cur = heap
        heap = heap.childlist
        cur.childlist = None

        # Unlink first path
        head = cur
        cur = cur.next
        head.next = None

        # Render path
        xor_path(bm, head)
        x0, y0, x1, y1 = setbbox_path(head)

        # Now do insideness test for each element of cur; append it to
        # head->childlist if it's inside head, else append it to head->next
        # (This would involve testing each path against the rendered bitmap)
    """


def bm_to_pathlist(
    bm: np.array, turdsize: int = 2, turnpolicy: int = POTRACE_TURNPOLICY_MINORITY
) -> list:
    """
    Decompose the given bitmap into paths.
    
    This is the main function for Stage 1 of the Potrace algorithm. It
    finds all black regions in the bitmap and traces their boundaries
    to create a list of path objects.
    
    Args:
        bm: The bitmap array to decompose
        turdsize: Minimum area for paths (smaller paths are ignored)
        turnpolicy: Strategy for resolving ambiguous turns during path tracing
        
    Returns:
        A list of _Path objects representing all the traced regions
        
    Note:
        Returns 0 on success with plistp set, or -1 on error with errno set.
        The byte padding on the right is set to 0, as the fast pixel search
        below relies on it.
    """
    plist = []  # List to store all found paths
    original = bm.copy()  # Keep original bitmap for reference

    # Iterate through all components (black regions) in the bitmap
    while True:
        # Find the next black pixel to start tracing from
        n = findnext(bm)
        if n is None:
            break  # No more black pixels found
            
        y, x = n
        
        # Calculate the sign by looking at the original bitmap
        # True = positive path (black region), False = negative path (white region)
        sign = original[y][x]
        
        # Calculate the path starting from the pixel boundary
        # Start at (x, y+1) which is the upper left corner of the pixel
        path = findpath(bm, x, y + 1, sign, turnpolicy)
        if path is None:
            raise ValueError("Failed to trace path")

        # Update the bitmap by removing the traced region
        xor_path(bm, path)

        # If the path is large enough (not a "turd"), add it to the list
        if path.area > turdsize:
            plist.append(path)

    # Note: The tree structure creation is commented out as it's not fully implemented
    # pathlist_to_tree(plist, original)
    
    return plist


# END DECOMPOSE SECTION.

# /* auxiliary functions */


def sign(x):
    """
    Return the sign of a number.
    
    Args:
        x: The number to determine the sign of
        
    Returns:
        1 for positive numbers, -1 for negative numbers, 0 for zero
    """
    if x > 0:
        return 1
    if x < 0:
        return -1
    else:
        return 0


def mod(a: int, n: int) -> int:
    """
    Modular arithmetic that works correctly for negative numbers.
    
    This function implements proper modular arithmetic that handles
    negative numbers correctly. The test for a>=n, while redundant,
    speeds up the mod function by 70% in the average case (significant
    since the program spends about 16% of its time here - or 40%
    without the test).
    
    Args:
        a: The number to take modulo of
        n: The modulus (must be positive)
        
    Returns:
        a mod n, properly handling negative values
    """
    return a % n if a >= n else a if a >= 0 else n - 1 - (-1 - a) % n


def floordiv(a: int, n: int):
    """
    Floor division that works correctly for negative numbers.
    
    The "floordiv" function returns the largest integer <= a/n,
    and again this works correctly for negative a, as long as
    a,n are integers and n>0.
    
    Args:
        a: The dividend
        n: The divisor (must be positive)
        
    Returns:
        The floor division result, properly handling negative values
    """
    return a // n if a >= 0 else -1 - (-1 - a) // n


def interval(t: float, a: _Point, b: _Point):
    """
    Calculate a point along the line segment from a to b.
    
    This function performs linear interpolation between two points.
    When t=0, returns point a; when t=1, returns point b.
    
    Args:
        t: Interpolation parameter (0.0 to 1.0)
        a: Starting point
        b: Ending point
        
    Returns:
        A point interpolated between a and b
    """
    return _Point(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y))


def dorth_infty(p0: _Point, p2: _Point):
    """
    Return a direction that is 90 degrees counterclockwise from p2-p0,
    but then restricted to one of the major wind directions (n, nw, w, etc).
    
    This function is used to find perpendicular directions for various
    geometric calculations in the algorithm.
    
    Args:
        p0: First point
        p2: Second point
        
    Returns:
        A point representing the perpendicular direction
    """
    return _Point(-sign(p2.y - p0.y), sign(p2.x - p0.x))


def dpara(p0: _Point, p1: _Point, p2: _Point) -> float:
    """
    Calculate the signed area of the parallelogram formed by three points.
    
    This function computes (p1-p0)x(p2-p0), which is the area of the
    parallelogram formed by the vectors p1-p0 and p2-p0. The sign
    indicates the orientation (clockwise vs counterclockwise).
    
    Args:
        p0: Base point
        p1: First vector endpoint
        p2: Second vector endpoint
        
    Returns:
        The signed area of the parallelogram
    """
    x1 = p1.x - p0.x
    y1 = p1.y - p0.y
    x2 = p2.x - p0.x
    y2 = p2.y - p0.y
    return x1 * y2 - x2 * y1


def ddenom(p0: _Point, p2: _Point) -> float:
    """
    Calculate the denominator for distance calculations.
    
    ddenom/dpara have the property that the square of radius 1 centered
    at p1 intersects the line p0p2 iff |dpara(p0,p1,p2)| <= ddenom(p0,p2).
    This is used for various geometric tests in the algorithm.
    
    Args:
        p0: First point defining the line
        p2: Second point defining the line
        
    Returns:
        The denominator value for distance calculations
    """
    r = dorth_infty(p0, p2)
    return r.y * (p2.x - p0.x) - r.x * (p2.y - p0.y)


def cyclic(a: int, b: int, c: int) -> int:
    """
    Check if three indices are in cyclic order.
    
    Returns 1 if a <= b < c < a, in a cyclic sense (mod n).
    This is used to check if indices wrap around a circular array.
    
    Args:
        a: First index
        b: Second index
        c: Third index
        
    Returns:
        1 if the indices are in cyclic order, 0 otherwise
    """
    if a <= c:
        return a <= b < c
    else:
        return a <= b or b < c


def pointslope(pp: _Path, i: int, j: int, ctr: _Point, dir: _Point) -> None:
    """
    Determine the center and slope of the line from point i to point j.
    
    This function calculates the center point and direction vector of
    the line segment from point i to point j. It assumes i<j and needs
    the "sum" components of the path to be precomputed.
    
    Args:
        pp: The path containing the points
        i: Starting point index
        j: Ending point index
        ctr: Output parameter for the center point
        dir: Output parameter for the direction vector
        
    Note:
        This function modifies the ctr and dir parameters in place.
    """
    # Assume i<j for the calculation
    n = len(pp)
    sums = pp._sums

    r = 0  # Number of rotations from i to j

    # Handle wraparound cases by adjusting indices
    while j >= n:
        j -= n
        r += 1

    while i >= n:
        i -= n
        r -= 1

    while j < 0:
        j += n
        r -= 1

    while i < 0:
        i += n
        r += 1

    # Calculate sums with wraparound handling
    x = sums[j + 1].x - sums[i].x + r * sums[n].x
    y = sums[j + 1].y - sums[i].y + r * sums[n].y
    x2 = sums[j + 1].x2 - sums[i].x2 + r * sums[n].x2
    xy = sums[j + 1].xy - sums[i].xy + r * sums[n].xy
    y2 = sums[j + 1].y2 - sums[i].y2 + r * sums[n].y2
    k = j + 1 - i + r * n

    # Calculate the center point
    ctr.x = x / k
    ctr.y = y / k

    # Calculate the covariance matrix
    a = float(x2 - x * x / k) / k
    b = float(xy - x * y / k) / k
    c = float(y2 - y * y / k) / k

    # Find the larger eigenvalue (principal component)
    lambda2 = (
        a + c + math.sqrt((a - c) * (a - c) + 4 * b * b)
    ) / 2

    # Calculate the corresponding eigenvector (direction)
    a -= lambda2
    c -= lambda2

    if math.fabs(a) >= math.fabs(c):
        l = math.sqrt(a * a + b * b)
        if l != 0:
            dir.x = -b / l
            dir.y = a / l
    else:
        l = math.sqrt(c * c + b * b)
        if l != 0:
            dir.x = -c / l
            dir.y = b / l
    
    # Handle the case where the eigenvalues coincide (degenerate case)
    if l == 0:
        # This can happen when k=4: the two eigenvalues coincide
        dir.x = 0
        dir.y = 0


# =============================================================================
# QUADRATIC FORMS AND VECTOR OPERATIONS
# =============================================================================
# These functions handle quadratic forms and various vector operations
# used in the curve fitting and optimization stages of the algorithm.

"""
The type of (affine) quadratic forms, represented as symmetric 3x3 matrices.
The value of the quadratic form at a vector (x,y) is v^t Q v, where v = (x,y,1)^t.
"""


def quadform(Q: list, w: _Point) -> float:
    """
    Apply quadratic form Q to vector w = (w.x,w.y).
    
    This function evaluates a quadratic form represented by a 3x3 matrix
    at a given 2D point. The quadratic form is used in various geometric
    calculations and optimizations.
    
    Args:
        Q: 3x3 symmetric matrix representing the quadratic form
        w: 2D point to evaluate the quadratic form at
        
    Returns:
        The value of the quadratic form at the given point
    """
    v = (w.x, w.y, 1.0)  # Homogeneous coordinates
    sum = 0.0
    for i in range(3):
        for j in range(3):
            sum += v[i] * Q[i][j] * v[j]
    return sum


def xprod(p1x, p1y, p2x, p2y) -> float:
    """
    Calculate the 2D cross product of two vectors.
    
    The cross product in 2D is a scalar that represents the signed area
    of the parallelogram formed by the two vectors.
    
    Args:
        p1x, p1y: Components of first vector
        p2x, p2y: Components of second vector
        
    Returns:
        The 2D cross product (scalar)
    """
    return p1x * p2y - p1y * p2x


def cprod(p0: _Point, p1: _Point, p2: _Point, p3: _Point) -> float:
    """
    Calculate the cross product of two line segments.
    
    This function computes (p1-p0)x(p3-p2), which is the cross product
    of the vectors from p0 to p1 and from p2 to p3.
    
    Args:
        p0, p1: First line segment endpoints
        p2, p3: Second line segment endpoints
        
    Returns:
        The cross product of the two line segments
    """
    x1 = p1.x - p0.x
    y1 = p1.y - p0.y
    x2 = p3.x - p2.x
    y2 = p3.y - p2.y
    return x1 * y2 - x2 * y1


def iprod(p0: _Point, p1: _Point, p2: _Point) -> float:
    """
    Calculate the dot product of two vectors from a common point.
    
    This function computes (p1-p0)*(p2-p0), which is the dot product
    of the vectors from p0 to p1 and from p0 to p2.
    
    Args:
        p0: Common base point
        p1, p2: Endpoints of the two vectors
        
    Returns:
        The dot product of the two vectors
    """
    x1 = p1.x - p0.x
    y1 = p1.y - p0.y
    x2 = p2.x - p0.x
    y2 = p2.y - p0.y
    return x1 * x2 + y1 * y2


def iprod1(p0: _Point, p1: _Point, p2: _Point, p3: _Point) -> float:
    """
    Calculate the dot product of two independent line segments.
    
    This function computes (p1-p0)*(p3-p2), which is the dot product
    of the vectors from p0 to p1 and from p2 to p3.
    
    Args:
        p0, p1: First line segment endpoints
        p2, p3: Second line segment endpoints
        
    Returns:
        The dot product of the two line segments
    """
    x1 = p1.x - p0.x
    y1 = p1.y - p0.y
    x2 = p3.x - p2.x
    y2 = p3.y - p2.y
    return x1 * x2 + y1 * y2


def sq(x: float) -> float:
    """
    Calculate the square of a number.
    
    Args:
        x: The number to square
        
    Returns:
        x squared
    """
    return x * x


def ddist(p: _Point, q: _Point) -> float:
    """
    Calculate the Euclidean distance between two points.
    
    Args:
        p, q: The two points
        
    Returns:
        The distance between the points
    """
    return math.sqrt(sq(p.x - q.x) + sq(p.y - q.y))


def bezier(t: float, p0: _Point, p1: _Point, p2: _Point, p3: _Point) -> _Point:
    """
    Calculate a point on a cubic Bezier curve.
    
    This function evaluates a cubic Bezier curve defined by four control points
    at parameter value t. The curve follows the standard Bezier formula:
    B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
    
    Args:
        t: Parameter value (0.0 to 1.0)
        p0, p1, p2, p3: The four control points of the Bezier curve
        
    Returns:
        The point on the Bezier curve at parameter t
        
    Note:
        A good optimizing compiler (such as gcc-3) reduces the following
        to 16 multiplications, using common subexpression elimination.
    """
    s = 1 - t  # Complementary parameter for optimization

    """
    Note: a good optimizing compiler (such as gcc-3) reduces the
    following to 16 multiplications, using common subexpression
    elimination.
    """
    return _Point(
        s * s * s * p0.x
        + 3 * (s * s * t) * p1.x
        + 3 * (t * t * s) * p2.x
        + t * t * t * p3.x,
        s * s * s * p0.y
        + 3 * (s * s * t) * p1.y
        + 3 * (t * t * s) * p2.y
        + t * t * t * p3.y,
    )


def tangent(
    p0: _Point, p1: _Point, p2: _Point, p3: _Point, q0: _Point, q1: _Point
) -> float:
    """
    Calculate the point t in [0..1] on the (convex) Bezier curve
    (p0,p1,p2,p3) which is tangent to q1-q0.
    
    This function finds the parameter value t where the Bezier curve
    is tangent to the line segment from q0 to q1. This is used in
    curve optimization to find intersection points.
    
    Args:
        p0, p1, p2, p3: The four control points of the Bezier curve
        q0, q1: The endpoints of the line segment to find tangency with
        
    Returns:
        The parameter value t in [0,1] where the curve is tangent to the line,
        or -1.0 if there is no solution in [0,1]
    """
    # The tangency condition leads to a quadratic equation:
    # (1-t)^2 A + 2(1-t)t B + t^2 C = 0
    # which can be rewritten as: a t^2 + b t + c = 0
    
    # Calculate the coefficients of the quadratic equation
    A = cprod(p0, p1, q0, q1)  # Cross product of first segment with line
    B = cprod(p1, p2, q0, q1)  # Cross product of second segment with line
    C = cprod(p2, p3, q0, q1)  # Cross product of third segment with line

    # Rewrite the equation in standard quadratic form
    a = A - 2 * B + C
    b = -2 * A + 2 * B
    c = A

    # Calculate the discriminant
    d = b * b - 4 * a * c

    # Check if there are real solutions
    if a == 0 or d < 0:
        return -1.0  # No real solutions

    # Calculate the two roots
    s = math.sqrt(d)
    r1 = (-b + s) / (2 * a)
    r2 = (-b - s) / (2 * a)

    # Return the root that lies in [0,1], or -1 if neither does
    if 0 <= r1 <= 1:
        return r1
    elif 0 <= r2 <= 1:
        return r2
    else:
        return -1.0


"""
/* ---------------------------------------------------------------------- */
/* Stage 1: determine the straight subpaths (Sec. 2.2.1). Fill in the
     "lon" component of a path object (based on pt/len).	For each i,
     lon[i] is the furthest index such that a straight line can be drawn
     from i to lon[i]. Return 1 on error with errno set, else 0. */

/* this algorithm depends on the fact that the existence of straight
     subpaths is a triplewise property. I.e., there exists a straight
     line through squares i0,...,in iff there exists a straight line
     through i,j,k, for all i0<=i<j<k<=in. (Proof?) */

/* this implementation of calc_lon is O(n^2). It replaces an older
     O(n^3) version. A "constraint" means that future points must
     satisfy xprod(constraint[0], cur) >= 0 and xprod(constraint[1],
     cur) <= 0. */

/* Remark for Potrace 1.1: the current implementation of calc_lon is
     more complex than the implementation found in Potrace 1.0, but it
     is considerably faster. The introduction of the "nc" data structure
     means that we only have to test the constraints for "corner"
     points. On a typical input file, this speeds up the calc_lon
     function by a factor of 31.2, thereby decreasing its time share
     within the overall Potrace algorithm from 72.6% to 7.82%, and
     speeding up the overall algorithm by a factor of 3.36. On another
     input file, calc_lon was sped up by a factor of 6.7, decreasing its
     time share from 51.4% to 13.61%, and speeding up the overall
     algorithm by a factor of 1.78. In any case, the savings are
     substantial. */

"""


# ----------------------------------------------------------------------


def _calc_sums(path: _Path) -> int:
    """
    Preparation: fill in the sum* fields of a path (used for later
    rapid summing).
    
    This function precomputes cumulative sums of coordinates and their
    products to enable fast calculations in later stages. These sums
    are used extensively in the polygon calculation and curve fitting
    stages.
    
    Args:
        path: The path object to prepare sums for
        
    Returns:
        0 on success, 1 with errno set on failure
    """
    n = len(path)
    path._sums = [_Sums() for i in range(len(path) + 1)]

    # Set the origin point for coordinate calculations
    path._x0 = path.pt[0].x
    path._y0 = path.pt[0].y

    # Initialize the first sum entry
    path._sums[0].x2 = 0
    path._sums[0].xy = 0
    path._sums[0].y2 = 0
    path._sums[0].x = 0
    path._sums[0].y = 0
    
    # Calculate cumulative sums for fast later calculations
    for i in range(n):
        # Convert to relative coordinates
        x = path.pt[i].x - path._x0
        y = path.pt[i].y - path._y0
        
        # Accumulate sums
        path._sums[i + 1].x = path._sums[i].x + x
        path._sums[i + 1].y = path._sums[i].y + y
        path._sums[i + 1].x2 = path._sums[i].x2 + float(x * x)
        path._sums[i + 1].xy = path._sums[i].xy + float(x * y)
        path._sums[i + 1].y2 = path._sums[i].y2 + float(y * y)
    
    return 0


def _calc_lon(pp: _Path) -> int:
    """
    Initialize the nc data structure and calculate longest straight lines.
    
    This function finds the longest straight line from each point to a future
    point. It uses an optimized algorithm that only tests constraints at
    "corner" points, making it much faster than the original O(n^3) version.
    
    The algorithm takes advantage of the fact that there is always a direction
    change at point 0 (due to the path decomposition algorithm). But even if
    this were not so, there is no harm, as in practice, correctness does not
    depend on the word "furthest" above.
    
    Args:
        pp: The path object to calculate longest lines for
        
    Returns:
        0 on success, 1 on error with errno set
    """
    pt = pp.pt
    n = len(pp)
    ct = [0, 0, 0, 0]  # Counter for each of the 4 directions
    pivk = [None] * n   # pivk[n]: pivot points
    nc = [None] * n     # nc[n]: next corner

    # Initialize the "next corner" array
    # For each point i, nc[i] points to the next corner after i
    k = 0
    for i in range(n - 1, -1, -1):
        if pt[i].x != pt[k].x and pt[i].y != pt[k].y:
            k = i + 1  # Necessarily i<n-1 in this case
        nc[i] = k

    pp._lon = [None] * n

    # Determine pivot points: for each i, let pivk[i] be the furthest k
    # such that all j with i<j<k lie on a line connecting i,k.
    for i in range(n - 1, -1, -1):
        # Reset direction counters
        ct[0] = ct[1] = ct[2] = ct[3] = 0

        # Keep track of "directions" that have occurred
        # Calculate the direction from point i to point i+1
        dir = int(
            (3 + 3 * (pt[mod(i + 1, n)].x - pt[i].x) + (pt[mod(i + 1, n)].y - pt[i].y))
            // 2
        )
        ct[dir] += 1

        # Initialize constraint variables
        constraint0x = 0
        constraint0y = 0
        constraint1x = 0
        constraint1y = 0

        # Find the next k such that no straight line from i to k
        k = nc[i]
        k1 = i
        while True:
            break_inner_loop_and_continue = False
            
            # Calculate direction from k1 to k
            dir = int(3 + 3 * sign(pt[k].x - pt[k1].x) + sign(pt[k].y - pt[k1].y)) // 2
            ct[dir] += 1

            # If all four "directions" have occurred, cut this path
            if ct[0] and ct[1] and ct[2] and ct[3]:
                pivk[i] = k1
                break_inner_loop_and_continue = True
                break  # goto foundk;

            # Calculate current vector from i to k
            cur_x = pt[k].x - pt[i].x
            cur_y = pt[k].y - pt[i].y

            # Check if current constraint is violated
            if (
                xprod(constraint0x, constraint0y, cur_x, cur_y) < 0
                or xprod(constraint1x, constraint1y, cur_x, cur_y) > 0
            ):
                break
                
            # Update constraint if not violated
            if abs(cur_x) <= 1 and abs(cur_y) <= 1:
                pass  # No constraint for very small vectors
            else:
                # Update constraint0 (first constraint)
                off_x = cur_x + (1 if (cur_y >= 0 and (cur_y > 0 or cur_x < 0)) else -1)
                off_y = cur_y + (1 if (cur_x <= 0 and (cur_x < 0 or cur_y < 0)) else -1)
                if xprod(constraint0x, constraint0y, off_x, off_y) >= 0:
                    constraint0x = off_x
                    constraint0y = off_y
                    
                # Update constraint1 (second constraint)
                off_x = cur_x + (1 if (cur_y <= 0 and (cur_y < 0 or cur_x < 0)) else -1)
                off_y = cur_y + (1 if (cur_x >= 0 and (cur_x > 0 or cur_y < 0)) else -1)
                if xprod(constraint1x, constraint1y, off_x, off_y) <= 0:
                    constraint1x = off_x
                    constraint1y = off_y
                    
            k1 = k
            k = nc[k1]
            if not cyclic(k, i, k1):
                break
                
        if break_inner_loop_and_continue:
            # This previously was a goto to the end of the for i statement.
            continue
            
        # Constraint violation handling:
        # k1 was the last "corner" satisfying the current constraint, and
        # k is the first one violating it. We now need to find the last
        # point along k1..k which satisfied the constraint.
        
        # Calculate direction of k-k1
        dk_x = sign(pt[k].x - pt[k1].x)
        dk_y = sign(pt[k].y - pt[k1].y)
        cur_x = pt[k1].x - pt[i].x
        cur_y = pt[k1].y - pt[i].y
        
        # Find largest integer j such that xprod(constraint[0], cur+j*dk) >= 0 
        # and xprod(constraint[1], cur+j*dk) <= 0. Use bilinearity of xprod.
        a = xprod(constraint0x, constraint0y, cur_x, cur_y)
        b = xprod(constraint0x, constraint0y, dk_x, dk_y)
        c = xprod(constraint1x, constraint1y, cur_x, cur_y)
        d = xprod(constraint1x, constraint1y, dk_x, dk_y)
        
        # Find largest integer j such that a+j*b>=0 and c+j*d<=0.
        # This can be solved with integer arithmetic using bilinearity of xprod.
        j = INFTY
        if b < 0:
            j = floordiv(a, -b)
        if d > 0:
            j = min(j, floordiv(-c, d))
        pivk[i] = mod(k1 + j, n)
        # foundk:
        # /* for i */

    # Clean up: for each i, let lon[i] be the largest k such that for
    # all i' with i<=i'<k, i'<k<=pivk[i'].
    # This creates the final longest line array from the pivot points.

    j = pivk[n - 1]
    pp._lon[n - 1] = j
    for i in range(n - 2, -1, -1):
        if cyclic(i + 1, pivk[i], j):
            j = pivk[i]
        pp._lon[i] = j

    # Final cleanup pass to ensure consistency
    i = n - 1
    while cyclic(mod(i + 1, n), j, pp._lon[i]):
        pp._lon[i] = j
        i -= 1
    return 0


# =============================================================================
# STAGE 2: CALCULATE THE OPTIMAL POLYGON
# =============================================================================
# Stage 2 of the Potrace algorithm: calculate the optimal polygon.
# This stage takes the longest line information from Stage 1 and
# finds the optimal polygon that approximates the path with the
# minimum number of vertices while maintaining accuracy.


def penalty3(pp: _Path, i: int, j: int) -> float:
    """
    Auxiliary function: calculate the penalty of an edge from i to j in
    the given path.
    
    This function calculates how well a straight line from point i to point j
    approximates the actual path between those points. The penalty is based
    on the squared distance between the line and the path points.
    
    This needs the "lon" and "sum*" data from Stage 1.
    
    Args:
        pp: The path object containing the path data
        i: Starting point index
        j: Ending point index
        
    Returns:
        The penalty value (lower is better)
    """
    n = len(pp)
    pt = pp.pt
    sums = pp._sums

    # Assume 0<=i<j<=n

    r = 0  # Rotations from i to j (handles wraparound)
    if j >= n:
        j -= n
        r = 1

    # Critical inner loop: the "if" gives a 4.6 percent speedup
    # Calculate sums with or without wraparound
    if r == 0:
        # No wraparound case
        x = sums[j + 1].x - sums[i].x
        y = sums[j + 1].y - sums[i].y
        x2 = sums[j + 1].x2 - sums[i].x2
        xy = sums[j + 1].xy - sums[i].xy
        y2 = sums[j + 1].y2 - sums[i].y2
        k = j + 1 - i
    else:
        # Wraparound case - add the full path sum
        x = sums[j + 1].x - sums[i].x + sums[n].x
        y = sums[j + 1].y - sums[i].y + sums[n].y
        x2 = sums[j + 1].x2 - sums[i].x2 + sums[n].x2
        xy = sums[j + 1].xy - sums[i].xy + sums[n].xy
        y2 = sums[j + 1].y2 - sums[i].y2 + sums[n].y2
        k = j + 1 - i + n

    # Calculate the midpoint of the line segment
    px = (pt[i].x + pt[j].x) / 2.0 - pt[0].x
    py = (pt[i].y + pt[j].y) / 2.0 - pt[0].y
    
    # Calculate the direction vector of the line segment
    ey = pt[j].x - pt[i].x
    ex = -(pt[j].y - pt[i].y)

    # Calculate the quadratic form coefficients
    # These represent the squared distance from the line to the path
    a = (x2 - 2 * x * px) / k + px * px
    b = (xy - x * py - y * px) / k + px * py
    c = (y2 - 2 * y * py) / k + py * py

    # Calculate the penalty as the square root of the quadratic form
    s = ex * ex * a + 2 * ex * ey * b + ey * ey * c
    return math.sqrt(s)


def _bestpolygon(pp: _Path) -> int:
    """
    Find the optimal polygon that approximates the path.
    
    This function uses dynamic programming to find the polygon with the
    minimum number of vertices that best approximates the original path.
    It fills in the m and po components of the path object.
    
    Non-cyclic version: assumes i=0 is in the polygon.
    TODO: implement cyclic version.
    
    Args:
        pp: The path object to find the optimal polygon for
        
    Returns:
        0 on success, 1 on failure with errno set
    """
    n = len(pp)
    pen = [None] * (n + 1)  # pen[n+1]: penalty vector
    prev = [None] * (n + 1)  # prev[n+1]: best path pointer vector
    clip0 = [None] * n  # clip0[n]: longest segment pointer, non-cyclic
    clip1 = [None] * (n + 1)  # clip1[n+1]: backwards segment pointer, non-cyclic
    seg0 = [None] * (n + 1)  # seg0[m+1]: forward segment bounds, m<=n
    seg1 = [None] * (n + 1)  # seg1[m+1]: backward segment bounds, m<=n

    # Calculate clipped paths
    # For each point i, find the furthest point that can be reached
    # by a straight line from i
    for i in range(n):
        c = mod(pp._lon[mod(i - 1, n)] - 1, n)
        if c == i:
            c = mod(i + 1, n)
        if c < i:
            clip0[i] = n  # Can't reach any point after i
        else:
            clip0[i] = c

    # Calculate backwards path clipping, non-cyclic.
    # j <= clip0[i] iff clip1[j] <= i, for i,j=0..n.
    # This creates a reverse mapping for efficient lookups
    j = 1
    for i in range(n):
        while j <= clip0[i]:
            clip1[j] = i
            j += 1

    # Calculate seg0[j] = longest path from 0 with j segments
    # This finds the maximum number of segments needed
    i = 0
    j = 0
    while i < n:
        seg0[j] = i
        i = clip0[i]
        j += 1
    seg0[j] = n
    m = j

    # Calculate seg1[j] = longest path to n with m-j segments
    # This creates the reverse path for dynamic programming
    i = n
    for j in range(m, 0, -1):
        seg1[j] = i
        i = clip1[i]
    seg1[0] = 0

    # Now find the shortest path with m segments, based on penalty3
    # Note: the outer 2 loops jointly have at most n iterations, thus
    # the worst-case behavior here is quadratic. In practice, it is
    # close to linear since the inner loop tends to be short.
    pen[0] = 0
    for j in range(1, m + 1):
        for i in range(seg1[j], seg0[j] + 1):
            best = -1
            # Try all possible previous points k
            for k in range(seg0[j - 1], clip1[i] - 1, -1):
                thispen = penalty3(pp, k, i) + pen[k]
                if best < 0 or thispen < best:
                    prev[i] = k
                    best = thispen
            pen[i] = best

    # Store the results
    pp._m = m
    pp._po = [None] * m

    # Read off shortest path by following the prev pointers backwards
    i = n
    j = m - 1
    while i > 0:
        i = prev[i]
        pp._po[j] = i
        j -= 1
    return 0


# =============================================================================
# STAGE 3: VERTEX ADJUSTMENT
# =============================================================================
# Stage 3 of the Potrace algorithm: vertex adjustment.
# This stage takes the optimal polygon from Stage 2 and adjusts
# the vertices to create smooth curves.


def _adjust_vertices(pp: _Path) -> int:
    """
    Adjust vertices of optimal polygon: calculate the intersection of
    the two "optimal" line segments, then move it into the unit square
    if it lies outside.
    
    This function takes the optimal polygon vertices and adjusts them
    to create smooth curves. It calculates the intersection of adjacent
    line segments and ensures the resulting points are within the unit
    square for numerical stability.
    
    Args:
        pp: The path object to adjust vertices for
        
    Returns:
        0 on success, 1 with errno set on error
    """
    m = pp._m
    po = pp._po
    n = len(pp)
    pt = pp.pt  # point_t
    x0 = pp._x0
    y0 = pp._y0

    ctr = [_Point() for i in range(m)]  # ctr[m]: center points
    dir = [_Point() for i in range(m)]  # dir[m]: direction vectors
    q = [
        [[0.0 for a in range(3)] for b in range(3)] for c in range(m)
    ]  # quadform_t q[m]: quadratic forms
    v = [0.0, 0.0, 0.0]  # Temporary vector
    s = _Point(0, 0)  # Temporary point
    pp._curve = _Curve(m)

    # Calculate "optimal" point-slope representation for each line segment
    # This creates the center and direction for each polygon edge
    for i in range(m):
        j = po[mod(i + 1, m)]
        j = mod(j - po[i], n) + po[i]
        pointslope(pp, po[i], j, ctr[i], dir[i])

        # Represent each line segment as a singular quadratic form;
        # the distance of a point (x,y) from the line segment will be
        # (x,y,1)Q(x,y,1)^t, where Q=q[i].
    for i in range(m):
        d = sq(dir[i].x) + sq(dir[i].y)
        if d == 0.0:
            for j in range(3):
                for k in range(3):
                    q[i][j][k] = 0
        else:
            v[0] = dir[i].y
            v[1] = -dir[i].x
            v[2] = -v[1] * ctr[i].y - v[0] * ctr[i].x
            for l in range(3):
                for k in range(3):
                    q[i][l][k] = v[l] * v[k] / d

    # Now calculate the "intersections" of consecutive segments.
    # Instead of using the actual intersection, we find the point
    # within a given unit square which minimizes the square distance to
    # the two lines.
    Q = [[0.0 for a in range(3)] for b in range(3)]
    for i in range(m):
        # Variables for minimum and candidate for minimum of quadratic form
        # Variables for coordinates of minimum

        # Let s be the vertex, in coordinates relative to x0/y0
        s.x = pt[po[i]].x - x0
        s.y = pt[po[i]].y - y0

        # Intersect segments i-1 and i
        j = mod(i - 1, m)

        # Add quadratic forms from both adjacent segments
        for l in range(3):
            for k in range(3):
                Q[l][k] = q[j][l][k] + q[i][l][k]

        while True:
            # Minimize the quadratic form Q on the unit square
            # Find intersection by solving the linear system

            det = Q[0][0] * Q[1][1] - Q[0][1] * Q[1][0]
            w = None
            if det != 0.0:
                # Matrix is non-singular, solve for intersection point
                w = _Point(
                    (-Q[0][2] * Q[1][1] + Q[1][2] * Q[0][1]) / det,
                    (Q[0][2] * Q[1][0] - Q[1][2] * Q[0][0]) / det,
                )
                break

            # Matrix is singular - lines are parallel. Add another,
            # orthogonal axis, through the center of the unit square
            if Q[0][0] > Q[1][1]:
                v[0] = -Q[0][1]
                v[1] = Q[0][0]
            elif Q[1][1]:
                v[0] = -Q[1][1]
                v[1] = Q[1][0]
            else:
                v[0] = 1
                v[1] = 0
            d = sq(v[0]) + sq(v[1])
            v[2] = -v[1] * s.y - v[0] * s.x
            for l in range(3):
                for k in range(3):
                    Q[l][k] += v[l] * v[k] / d
                    
        # Check if the intersection point is within the unit square
        dx = math.fabs(w.x - s.x)
        dy = math.fabs(w.y - s.y)
        if dx <= 0.5 and dy <= 0.5:
            # Point is within unit square, use it directly
            pp._curve[i].vertex.x = w.x + x0
            pp._curve[i].vertex.y = w.y + y0
            continue

        # The minimum was not in the unit square; now minimize quadratic
        # on boundary of square
        min = quadform(Q, s)
        xmin = s.x
        ymin = s.y

        # Check the four edges of the unit square
        if Q[0][0] != 0.0:
            for z in range(2):  # Value of the y-coordinate
                w.y = s.y - 0.5 + z
                w.x = -(Q[0][1] * w.y + Q[0][2]) / Q[0][0]
                dx = math.fabs(w.x - s.x)
                cand = quadform(Q, w)
                if dx <= 0.5 and cand < min:
                    min = cand
                    xmin = w.x
                    ymin = w.y
        if Q[1][1] != 0.0:
            for z in range(2):  # Value of the x-coordinate
                w.x = s.x - 0.5 + z
                w.y = -(Q[1][0] * w.x + Q[1][2]) / Q[1][1]
                dy = math.fabs(w.y - s.y)
                cand = quadform(Q, w)
                if dy <= 0.5 and cand < min:
                    min = cand
                    xmin = w.x
                    ymin = w.y
                    
        # Check four corners of the unit square
        for l in range(2):
            for k in range(2):
                w = _Point(s.x - 0.5 + l, s.y - 0.5 + k)
                cand = quadform(Q, w)
                if cand < min:
                    min = cand
                    xmin = w.x
                    ymin = w.y
                    
        # Store the adjusted vertex
        pp._curve[i].vertex.x = xmin + x0
        pp._curve[i].vertex.y = ymin + y0
    return 0


# =============================================================================
# STAGE 4: SMOOTHING AND CORNER ANALYSIS
# =============================================================================
# Stage 4 of the Potrace algorithm: smoothing and corner analysis.
# This stage takes the adjusted vertices from Stage 3 and determines
# which vertices should be sharp corners and which should be smooth curves.


def reverse(curve: _Curve) -> None:
    """
    Reverse the orientation of a path.
    
    This function swaps the order of vertices in the curve, effectively
    reversing the direction of the path. This is used for negative paths
    (white regions) to ensure consistent orientation.
    
    Args:
        curve: The curve to reverse
    """
    m = curve.n
    i = 0
    j = m - 1
    while i < j:
        # Swap vertices at positions i and j
        tmp = curve[i].vertex
        curve[i].vertex = curve[j].vertex
        curve[j].vertex = tmp
        i += 1
        j -= 1


def _smooth(curve: _Curve, alphamax: float) -> None:
    """
    Smooth the curve by analyzing each vertex and determining whether
    it should be a sharp corner or a smooth curve.
    
    This function examines each vertex and calculates a smoothness parameter
    (alpha) based on the local geometry. Vertices with high alpha become
    sharp corners, while those with low alpha become smooth curve segments.
    
    Args:
        curve: The curve to smooth
        alphamax: Maximum smoothness parameter (0.0 to 1.0)
                 Higher values create more sharp corners
    """
    m = curve.n

    # Examine each vertex and find its best fit
    for i in range(m):
        j = mod(i + 1, m)  # Next vertex
        k = mod(i + 2, m)  # Vertex after next
        p4 = interval(1 / 2.0, curve[k].vertex, curve[j].vertex)

        # Calculate the smoothness parameter alpha
        denom = ddenom(curve[i].vertex, curve[k].vertex)
        if denom != 0.0:
            dd = dpara(curve[i].vertex, curve[j].vertex, curve[k].vertex) / denom
            dd = math.fabs(dd)
            alpha = (1 - 1.0 / dd) if dd > 1 else 0
            alpha = alpha / 0.75
        else:
            alpha = 4 / 3.0
        curve[j].alpha0 = alpha  # Remember "original" value of alpha

        if alpha >= alphamax:  # Pointed corner
            # Create a sharp corner segment
            curve[j].tag = POTRACE_CORNER
            curve[j].c[1] = curve[j].vertex
            curve[j].c[2] = p4
        else:
            # Create a smooth curve segment
            if alpha < 0.55:
                alpha = 0.55
            elif alpha > 1:
                alpha = 1
            p2 = interval(0.5 + 0.5 * alpha, curve[i].vertex, curve[j].vertex)
            p3 = interval(0.5 + 0.5 * alpha, curve[k].vertex, curve[j].vertex)
            curve[j].tag = POTRACE_CURVETO
            curve[j].c[0] = p2
            curve[j].c[1] = p3
            curve[j].c[2] = p4
        curve[j].alpha = alpha  # Store the "cropped" value of alpha
        curve[j].beta = 0.5
    curve.alphacurve = True


# =============================================================================
# STAGE 5: CURVE OPTIMIZATION
# =============================================================================
# Stage 5 of the Potrace algorithm: curve optimization.
# This stage takes the smoothed curves from Stage 4 and optimizes them
# by combining multiple Bezier segments into single segments where possible.


class opti_t:
    """
    Optimization result container.
    
    This class holds the results of curve optimization calculations,
    including penalty values and curve parameters.
    """
    
    def __init__(self):
        self.pen = 0  # Penalty value (lower is better)
        self.c = [_Point(0, 0), _Point(0, 0)]  # Curve control points
        self.t = 0  # Curve parameter t
        self.s = 0  # Curve parameter s
        self.alpha = 0  # Curve parameter alpha


def opti_penalty(
    pp: _Path,
    i: int,
    j: int,
    res: opti_t,
    opttolerance: float,
    convc: int,
    areac: float,
) -> int:
    """
    Calculate best fit from i+.5 to j+.5. Assume i<j (cyclically).
    
    This function attempts to find the optimal Bezier curve that fits
    the path from point i+0.5 to point j+0.5. It checks various
    constraints including convexity, corner-freeness, and maximum bend.
    
    Args:
        pp: The path object
        i: Starting point index
        j: Ending point index
        res: Output parameter for optimization results
        opttolerance: Tolerance for optimization
        convc: Pre-computed convexity array
        areac: Pre-computed area array
        
    Returns:
        0 if optimization is possible and results are set in res,
        1 if optimization is impossible
    """
    m = pp._curve.n

    # Check convexity, corner-freeness, and maximum bend < 179 degrees
    if i == j:  # Sanity check - a full loop can never be an opticurve
        return 1

    k = i
    i1 = mod(i + 1, m)
    k1 = mod(k + 1, m)
    conv = convc[k1]
    if conv == 0:
        return 1
    d = ddist(pp._curve[i].vertex, pp._curve[i1].vertex)
    k = k1
    while k != j:
        k1 = mod(k + 1, m)
        k2 = mod(k + 2, m)
        if convc[k1] != conv:
            return 1
        if (
            sign(
                cprod(
                    pp._curve[i].vertex,
                    pp._curve[i1].vertex,
                    pp._curve[k1].vertex,
                    pp._curve[k2].vertex,
                )
            )
            != conv
        ):
            return 1
        if (
            iprod1(
                pp._curve[i].vertex,
                pp._curve[i1].vertex,
                pp._curve[k1].vertex,
                pp._curve[k2].vertex,
            )
            < d * ddist(pp._curve[k1].vertex, pp._curve[k2].vertex) * COS179
        ):
            return 1
        k = k1

    # /* the curve we're working in: */
    p0 = pp._curve[mod(i, m)].c[2]
    p1 = pp._curve[mod(i + 1, m)].vertex
    p2 = pp._curve[mod(j, m)].vertex
    p3 = pp._curve[mod(j, m)].c[2]

    # /* determine its area */
    area = areac[j] - areac[i]
    area -= dpara(pp._curve[0].vertex, pp._curve[i].c[2], pp._curve[j].c[2]) / 2
    if i >= j:
        area += areac[m]

    # /* find intersection o of p0p1 and p2p3. Let t,s such that
    # o =interval(t,p0,p1) = interval(s,p3,p2). Let A be the area of the
    # triangle (p0,o,p3). */

    A1 = dpara(p0, p1, p2)
    A2 = dpara(p0, p1, p3)
    A3 = dpara(p0, p2, p3)
    # /* A4 = dpara(p1, p2, p3); */
    A4 = A1 + A3 - A2

    if A2 == A1:  # this should never happen
        return 1

    t = A3 / (A3 - A4)
    s = A2 / (A2 - A1)
    A = A2 * t / 2.0

    if A == 0.0:  # this should never happen
        return 1

    R = area / A  # /* relative area */
    alpha = 2 - math.sqrt(4 - R / 0.3)  # /* overall alpha for p0-o-p3 curve */

    res.c[0] = interval(t * alpha, p0, p1)
    res.c[1] = interval(s * alpha, p3, p2)
    res.alpha = alpha
    res.t = t
    res.s = s

    p1 = res.c[0]
    p1 = _Point(p1.x, p1.y)
    p2 = res.c[1]  # The proposed curve is now (p0,p1,p2,p3)
    p2 = _Point(p2.x, p2.y)

    res.pen = 0

    # Calculate penalty by checking tangency with edges
    k = mod(i + 1, m)
    while k != j:
        k1 = mod(k + 1, m)
        t = tangent(p0, p1, p2, p3, pp._curve[k].vertex, pp._curve[k1].vertex)
        if t < -0.5:
            return 1
        pt = bezier(t, p0, p1, p2, p3)
        d = ddist(pp._curve[k].vertex, pp._curve[k1].vertex)
        if d == 0.0:  # This should never happen
            return 1
        d1 = dpara(pp._curve[k].vertex, pp._curve[k1].vertex, pt) / d
        if math.fabs(d1) > opttolerance:
            return 1
        if (
            iprod(pp._curve[k].vertex, pp._curve[k1].vertex, pt) < 0
            or iprod(pp._curve[k1].vertex, pp._curve[k].vertex, pt) < 0
        ):
            return 1
        res.pen += sq(d1)
        k = k1

    # Check corners for additional penalty
    k = i
    while k != j:
        k1 = mod(k + 1, m)
        t = tangent(p0, p1, p2, p3, pp._curve[k].c[2], pp._curve[k1].c[2])
        if t < -0.5:
            return 1
        pt = bezier(t, p0, p1, p2, p3)
        d = ddist(pp._curve[k].c[2], pp._curve[k1].c[2])
        if d == 0.0:  # This should never happen
            return 1
        d1 = dpara(pp._curve[k].c[2], pp._curve[k1].c[2], pt) / d
        d2 = dpara(pp._curve[k].c[2], pp._curve[k1].c[2], pp._curve[k1].vertex) / d
        d2 *= 0.75 * pp._curve[k1].alpha
        if d2 < 0:
            d1 = -d1
            d2 = -d2
        if d1 < d2 - opttolerance:
            return 1
        if d1 < d2:
            res.pen += sq(d1 - d2)
        k = k1
    return 0


def _opticurve(pp: _Path, opttolerance: float) -> int:
    """
    Optimize the path p, replacing sequences of Bezier segments by a
    single segment when possible.
    
    This function uses dynamic programming to find the optimal combination
    of curve segments that minimizes the total penalty while maintaining
    accuracy within the specified tolerance.
    
    Args:
        pp: The path object to optimize
        opttolerance: Tolerance for optimization
        
    Returns:
        0 on success, 1 with errno set on failure
    """
    m = pp._curve.n
    pt = [0] * (m + 1)  # pt[m+1]: path tracking array
    pen = [0.0] * (m + 1)  # pen[m+1]: penalty array
    len = [0] * (m + 1)  # len[m+1]: length array
    opt = [None] * (m + 1)  # opt[m+1]: optimization results

    convc = [0.0] * m  # conv[m]: pre-computed convexities
    areac = [0.0] * (m + 1)  # cumarea[m+1]: cache for fast area computation

    # Pre-calculate convexity: +1 = right turn, -1 = left turn, 0 = corner
    for i in range(m):
        if pp._curve[i].tag == POTRACE_CURVETO:
            convc[i] = sign(
                dpara(
                    pp._curve[mod(i - 1, m)].vertex,
                    pp._curve[i].vertex,
                    pp._curve[mod(i + 1, m)].vertex,
                )
            )
        else:
            convc[i] = 0

    # Pre-calculate areas for fast computation
    area = 0.0
    areac[0] = 0.0
    p0 = pp._curve[0].vertex
    for i in range(m):
        i1 = mod(i + 1, m)
        if pp._curve[i1].tag == POTRACE_CURVETO:
            alpha = pp._curve[i1].alpha
            area += (
                0.3
                * alpha
                * (4 - alpha)
                * dpara(pp._curve[i].c[2], pp._curve[i1].vertex, pp._curve[i1].c[2])
                / 2
            )
            area += dpara(p0, pp._curve[i].c[2], pp._curve[i1].c[2]) / 2
        areac[i + 1] = area
        
    # Initialize dynamic programming arrays
    pt[0] = -1
    pen[0] = 0
    len[0] = 0

    # TODO: We always start from a fixed point
    # -- should find the best curve cyclically

    o = None
    for j in range(1, m + 1):
        # Calculate best path from 0 to j
        pt[j] = j - 1
        pen[j] = pen[j - 1]
        len[j] = len[j - 1] + 1
        for i in range(j - 2, -1, -1):
            if o is None:
                o = opti_t()
            if opti_penalty(pp, i, mod(j, m), o, opttolerance, convc, areac):
                break
            if len[j] > len[i] + 1 or (
                len[j] == len[i] + 1 and pen[j] > pen[i] + o.pen
            ):
                opt[j] = o
                pt[j] = i
                pen[j] = pen[i] + o.pen
                len[j] = len[i] + 1
                o = None
                
    # Create the optimized curve
    om = len[m]
    pp._ocurve = _Curve(om)
    s = [None] * om
    t = [None] * om

    # Reconstruct the optimal path
    j = m
    for i in range(om - 1, -1, -1):
        if pt[j] == j - 1:
            # Copy original segment
            pp._ocurve[i].tag = pp._curve[mod(j, m)].tag
            pp._ocurve[i].c[0] = pp._curve[mod(j, m)].c[0]
            pp._ocurve[i].c[1] = pp._curve[mod(j, m)].c[1]
            pp._ocurve[i].c[2] = pp._curve[mod(j, m)].c[2]
            pp._ocurve[i].vertex = pp._curve[mod(j, m)].vertex
            pp._ocurve[i].alpha = pp._curve[mod(j, m)].alpha
            pp._ocurve[i].alpha0 = pp._curve[mod(j, m)].alpha0
            pp._ocurve[i].beta = pp._curve[mod(j, m)].beta
            s[i] = t[i] = 1.0
        else:
            # Use optimized segment
            pp._ocurve[i].tag = POTRACE_CURVETO
            pp._ocurve[i].c[0] = opt[j].c[0]
            pp._ocurve[i].c[1] = opt[j].c[1]
            pp._ocurve[i].c[2] = pp._curve[mod(j, m)].c[2]
            pp._ocurve[i].vertex = interval(
                opt[j].s, pp._curve[mod(j, m)].c[2], pp._curve[mod(j, m)].vertex
            )
            pp._ocurve[i].alpha = opt[j].alpha
            pp._ocurve[i].alpha0 = opt[j].alpha
            s[i] = opt[j].s
            t[i] = opt[j].t
        j = pt[j]

    # Calculate beta parameters for smooth transitions
    for i in range(om):
        i1 = mod(i + 1, om)
        pp._ocurve[i].beta = s[i] / (s[i] + t[i1])
    pp._ocurve.alphacurve = True
    return 0


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================
# This function orchestrates the entire 5-stage Potrace algorithm
# for a list of paths, applying all stages in sequence.


def process_path(
    plist: list,
    alphamax=1.0,
    opticurve=True,
    opttolerance=0.2,
) -> int:
    """
    Process a list of paths through all 5 stages of the Potrace algorithm.
    
    This is the main orchestrating function that applies all stages of the
    Potrace algorithm to each path in the list. It handles errors gracefully
    and ensures all paths are processed completely.
    
    Args:
        plist: List of path objects to process
        alphamax: Maximum smoothness parameter for Stage 4
        opticurve: Whether to enable curve optimization in Stage 5
        opttolerance: Tolerance for curve optimization
        
    Returns:
        0 on success, 1 on error with errno set
    """

    def TRY(x):
        """Helper function to check for errors and raise exceptions."""
        if x:
            raise ValueError

    # Call downstream function with each path
    for p in plist:
        # Stage 1: Calculate sums for fast computations
        TRY(_calc_sums(p))
        
        # Stage 1: Calculate longest straight lines
        TRY(_calc_lon(p))
        
        # Stage 2: Find optimal polygon
        TRY(_bestpolygon(p))
        
        # Stage 3: Adjust vertices for smooth curves
        TRY(_adjust_vertices(p))
        
        # Reverse orientation of negative paths (white regions)
        if not p.sign:
            reverse(p._curve)
            
        # Stage 4: Smooth curves and determine corners
        _smooth(p._curve, alphamax)
        
        # Stage 5: Optimize curves (optional)
        if opticurve:
            TRY(_opticurve(p, opttolerance))
            p._fcurve = p._ocurve
        else:
            p._fcurve = p._curve
            
    return 0


# =============================================================================
# END OF POTRACE ALGORITHM IMPLEMENTATION
# =============================================================================
# This completes the Python implementation of the Potrace algorithm.
# The algorithm converts bitmap images into smooth vector curves through
# a 5-stage process: path decomposition, polygon optimization, vertex
# adjustment, smoothing, and curve optimization.
