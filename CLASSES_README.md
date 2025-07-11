# Potrace Classes Documentation

This document provides a comprehensive overview of all classes in the Potrace Python implementation. The classes are organized by their role in the algorithm pipeline.

## Table of Contents

1. [Public API Classes](#public-api-classes)
2. [Segment Classes](#segment-classes)
3. [Internal Data Structures](#internal-data-structures)
4. [Utility Classes](#utility-classes)
5. [Class Relationships](#class-relationships)
6. [Usage Examples](#usage-examples)

---

## Public API Classes

These classes form the main interface for users of the Potrace library.

### Bitmap Class

**Purpose**: Main entry point for bitmap processing and vectorization.

**Location**: `potrace/potrace.py` (lines 50-145)

**Key Methods**:
- `__init__(data, blacklevel=0.5)`: Initialize from various input formats
- `invert()`: Swap black and white pixels
- `trace(turdsize, turnpolicy, alphamax, opticurve, opttolerance)`: Main tracing method

**Input Formats Supported**:
- NumPy arrays (2D boolean or numeric)
- PIL Image objects
- Any array-like object

**Example Usage**:
```python
import numpy as np
from potrace import Bitmap

# Create bitmap from numpy array
data = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
bitmap = Bitmap(data)

# Trace to vector paths
paths = bitmap.trace()
```

**Properties**:
- `data`: The processed bitmap array (numpy.ndarray)

---

### Path Class

**Purpose**: Container for multiple curves forming a complete vector path.

**Location**: `potrace/potrace.py` (lines 146-185)

**Inheritance**: Inherits from `list` for easy iteration

**Key Properties**:
- `curves`: List of Curve objects (same as self)
- `curves_tree`: Hierarchical structure (not implemented)

**Example Usage**:
```python
# Path is returned by Bitmap.trace()
paths = bitmap.trace()

# Iterate over curves
for curve in paths:
    print(f"Curve has {len(curve)} segments")

# Access curves property
curves = paths.curves
```

---

### Curve Class

**Purpose**: Represents a single closed curve with multiple segments.

**Location**: `potrace/potrace.py` (lines 186-247)

**Inheritance**: Inherits from `list` for easy iteration over segments

**Key Properties**:
- `decomposition_points`: Original path points from bitmap decomposition
- `segments`: List of segments (same as self)
- `children`: Child curves (not implemented)
- `start_point`: Starting point of the curve

**Segment Types**:
- `CornerSegment`: Straight line segments
- `BezierSegment`: Smooth curve segments

**Example Usage**:
```python
for curve in paths:
    # Access original bitmap points
    points = curve.decomposition_points
    
    # Iterate over segments
    for segment in curve:
        if segment.is_corner:
            print(f"Corner at {segment.c}")
        else:
            print(f"Bezier curve: {segment.c1} -> {segment.c2}")
```

---

## Segment Classes

These classes represent individual segments within a curve.

### CornerSegment Class

**Purpose**: Represents a straight line segment (sharp corner) in a curve.

**Location**: `potrace/potrace.py` (lines 248-285)

**Key Properties**:
- `c`: Corner point (vertex)
- `end_point`: End point of the segment
- `is_corner`: Always returns `True`

**Example Usage**:
```python
for segment in curve:
    if segment.is_corner:
        # This is a CornerSegment
        corner_point = segment.c
        end_point = segment.end_point
```

---

### BezierSegment Class

**Purpose**: Represents a Bezier curve segment (smooth curve) in a curve.

**Location**: `potrace/potrace.py` (lines 286-335)

**Key Properties**:
- `c1`: First control point of the Bezier curve
- `c2`: Second control point of the Bezier curve
- `end_point`: End point of the Bezier curve
- `is_corner`: Always returns `False`

**Bezier Curve Structure**:
- Control points: `c1`, `c2`
- End point: `end_point`
- The curve follows the standard cubic Bezier formula

**Example Usage**:
```python
for segment in curve:
    if not segment.is_corner:
        # This is a BezierSegment
        control1 = segment.c1
        control2 = segment.c2
        end = segment.end_point
```

---

## Internal Data Structures

These classes are used internally during the processing pipeline and are not part of the public API.

### _Curve Class

**Purpose**: Internal curve representation used during processing.

**Location**: `potrace/potrace.py` (lines 336-365)

**Key Properties**:
- `segments`: List of _Segment objects
- `alphacurve`: Whether curve has been smoothed
- `n`: Number of segments (property)

**Methods**:
- `__len__()`: Return number of segments
- `__getitem__(item)`: Allow indexing to access segments

**Usage**: Used internally by the algorithm stages

---

### _Path Class

**Purpose**: Internal path representation used during processing.

**Location**: `potrace/potrace.py` (lines 366-420)

**Key Properties**:
- `pt`: List of points defining the path boundary
- `area`: Area enclosed by the path
- `sign`: Whether path is positive (True) or negative (False)

**Processing Data**:
- Stage 1: `_lon` (longest lines)
- Stage 2: `_x0`, `_y0`, `_sums` (coordinate calculations)
- Stage 3-5: `_m`, `_po`, `_curve`, `_ocurve`, `_fcurve` (curve data)

**Tree Structure** (not fully implemented):
- `next`: Next path in linked list
- `childlist`: Child paths
- `sibling`: Sibling paths

---

### _Point Class

**Purpose**: Simple 2D point representation used throughout the algorithm.

**Location**: `potrace/potrace.py` (lines 421-441)

**Properties**:
- `x`: X coordinate
- `y`: Y coordinate

**Methods**:
- `__repr__()`: String representation

**Example Usage**:
```python
point = _Point(10.5, 20.3)
print(point)  # Output: Point(10.500000, 20.300000)
```

---

### _Segment Class

**Purpose**: Internal segment representation used during curve processing.

**Location**: `potrace/potrace.py` (lines 442-460)

**Key Properties**:
- `tag`: Type (POTRACE_CORNER or POTRACE_CURVETO)
- `c`: Control points for Bezier curves (list of 3 _Point objects)
- `vertex`: Vertex point of the segment
- `alpha`: Smoothness parameter (0.0 to 1.0)
- `alpha0`: Original alpha value before optimization
- `beta`: Curve parameter for optimization

**Usage**: Used internally by the algorithm stages

---

### _Sums Class

**Purpose**: Container for precomputed sums used in fast calculations.

**Location**: `potrace/potrace.py` (lines 461-475)

**Properties**:
- `x`: Sum of x coordinates
- `y`: Sum of y coordinates
- `x2`: Sum of x² coordinates
- `xy`: Sum of x*y coordinates
- `y2`: Sum of y² coordinates

**Usage**: Used in Stage 2 for polygon calculations

---

## Utility Classes

### opti_t Class

**Purpose**: Optimization result container for curve optimization.

**Location**: `potrace/potrace.py` (lines 2109-2124)

**Properties**:
- `pen`: Penalty value (lower is better)
- `c`: Curve control points (list of 2 _Point objects)
- `t`: Curve parameter t
- `s`: Curve parameter s
- `alpha`: Curve parameter alpha

**Usage**: Used internally by the `opti_penalty` function

---

## Class Relationships

### Hierarchy Overview

```
Bitmap
└── trace() → Path
    └── contains multiple Curve objects
        └── contains multiple Segment objects
            ├── CornerSegment (straight lines)
            └── BezierSegment (smooth curves)
```

### Internal Processing Pipeline

```
_Path (internal)
├── Stage 1: _calc_sums(), _calc_lon()
├── Stage 2: _bestpolygon()
├── Stage 3: _adjust_vertices()
├── Stage 4: _smooth()
└── Stage 5: _opticurve()
    └── opti_t (optimization results)
```

### Data Flow

1. **Input**: Bitmap class receives image data
2. **Decomposition**: Creates _Path objects for each region
3. **Processing**: Each _Path goes through 5 stages
4. **Output**: _Path objects converted to public Curve objects
5. **Final Result**: Path objects containing Curve objects with Segment objects

---

## Usage Examples

### Basic Usage

```python
from potrace import Bitmap
import numpy as np

# Create a simple bitmap
data = np.array([
    [0, 1, 1, 0],
    [1, 1, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 0]
])

# Process the bitmap
bitmap = Bitmap(data)
paths = bitmap.trace()

# Access the results
for path in paths:
    for curve in path:
        print(f"Curve with {len(curve)} segments")
        for segment in curve:
            if segment.is_corner:
                print(f"  Corner at {segment.c}")
            else:
                print(f"  Bezier: {segment.c1} -> {segment.c2}")
```

### Advanced Usage

```python
# Custom parameters
paths = bitmap.trace(
    turdsize=5,           # Minimum area for paths
    turnpolicy=2,         # Always turn left when ambiguous
    alphamax=0.8,         # More sharp corners
    opticurve=True,       # Enable curve optimization
    opttolerance=0.1      # Stricter optimization tolerance
)

# Access original decomposition points
for path in paths:
    for curve in path:
        original_points = curve.decomposition_points
        print(f"Original bitmap points: {len(original_points)}")
```

### Working with Segments

```python
for path in paths:
    for curve in path:
        # Get the starting point
        start = curve.start_point
        
        # Process each segment
        for i, segment in enumerate(curve):
            if segment.is_corner:
                # Handle straight line segment
                corner_point = segment.c
                end_point = segment.end_point
                print(f"Segment {i}: Straight line to {end_point}")
            else:
                # Handle Bezier curve segment
                control1 = segment.c1
                control2 = segment.c2
                end_point = segment.end_point
                print(f"Segment {i}: Bezier curve to {end_point}")
```

---

## Design Patterns

### 1. Pipeline Pattern
The algorithm follows a 5-stage pipeline where each stage processes the output of the previous stage.

### 2. Strategy Pattern
Different turn policies and optimization strategies can be selected via parameters.

### 3. Composite Pattern
Path contains multiple Curve objects, which contain multiple Segment objects.

### 4. Factory Pattern
The Bitmap class acts as a factory, creating Path objects from bitmap data.

### 5. Iterator Pattern
All container classes (Path, Curve) inherit from list, providing easy iteration.

---

## Performance Considerations

### Memory Usage
- Internal classes (_Path, _Curve, _Segment) are created during processing
- Public classes (Path, Curve, CornerSegment, BezierSegment) are created at the end
- Large bitmaps may create many internal objects

### Processing Time
- Stage 1: O(n²) where n is the number of path points
- Stage 2: O(n²) dynamic programming
- Stage 3: O(m) where m is the number of polygon vertices
- Stage 4: O(m) smoothing
- Stage 5: O(m²) curve optimization

### Optimization Tips
- Use `turdsize` to filter out small paths
- Adjust `alphamax` to control corner vs curve classification
- Use `opticurve=False` to skip curve optimization for faster processing

---

## Error Handling

### Common Issues
1. **Empty bitmap**: No black pixels found
2. **Invalid parameters**: Out-of-range values for alphamax, opttolerance
3. **Memory errors**: Very large bitmaps may cause memory issues

### Exception Types
- `ValueError`: Invalid parameters or processing errors
- `IndexError`: Array bounds issues (rare)
- `MemoryError`: Insufficient memory for large bitmaps

---

## Extension Points

### Custom Turn Policies
The algorithm supports custom turn policies by implementing the turn resolution logic.

### Custom Optimization
The `opti_t` class can be extended for custom curve optimization strategies.

### Custom Segment Types
New segment types can be added by extending the segment base classes.

---

This documentation provides a complete overview of all classes in the Potrace implementation, their relationships, and usage patterns. The classes work together to convert bitmap images into smooth vector curves through a sophisticated 5-stage algorithm. 