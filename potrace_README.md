# Potrace Python Implementation

## Overview

This is a Python implementation of the Potrace algorithm, a powerful tool for tracing bitmap images into vector graphics. The algorithm converts raster images (bitmaps) into scalable vector paths, making it ideal for converting scanned drawings, logos, or any bitmap image into editable vector format.

## What is Potrace?

Potrace is an algorithm that:
- Takes a bitmap (black and white image) as input
- Analyzes the image to find closed paths around black regions
- Converts these paths into smooth vector curves
- Outputs scalable vector graphics that can be used in design software

## Architecture Overview

The implementation is organized into several key components:

### 1. Core Classes

#### `Bitmap`
- **Purpose**: Represents and processes input bitmap images
- **Key Methods**:
  - `__init__()`: Converts various input formats (numpy arrays, PIL images) to boolean bitmap
  - `invert()`: Inverts the bitmap (swaps black/white)
  - `trace()`: Main entry point for tracing the bitmap to vector paths

#### `Path`
- **Purpose**: Container for multiple curves that form a complete vector path
- **Properties**:
  - `curves`: List of individual curves
  - `curves_tree`: (Currently unimplemented) Hierarchical structure of curves

#### `Curve`
- **Purpose**: Represents a single closed curve with multiple segments
- **Properties**:
  - `segments`: List of curve segments (corners or Bezier curves)
  - `decomposition_points`: Original path points
  - `children`: (Currently unimplemented) Child curves

#### `CornerSegment` and `BezierSegment`
- **Purpose**: Represent individual segments of a curve
- **CornerSegment**: Straight line segments
- **BezierSegment**: Curved segments with control points

### 2. Internal Data Structures

#### `_Path`
- **Purpose**: Internal representation of a path during processing
- **Key Components**:
  - `pt`: List of points defining the path
  - `area`: Area enclosed by the path
  - `sign`: Whether path is positive (black) or negative (white)
  - `_lon`: Longest straight line information
  - `_curve`: Final curve representation

#### `_Curve`
- **Purpose**: Internal curve representation during processing
- **Components**:
  - `segments`: List of curve segments
  - `alphacurve`: Whether curve has been smoothed

#### `_Point`
- **Purpose**: Simple 2D point representation
- **Properties**: `x`, `y` coordinates

#### `_Segment`
- **Purpose**: Internal segment representation
- **Components**:
  - `tag`: Type of segment (corner or curve)
  - `c`: Control points for Bezier curves
  - `vertex`: Vertex point
  - `alpha`, `beta`: Curve parameters

## Algorithm Stages

The Potrace algorithm works in several distinct stages:

### Stage 1: Path Decomposition
**Files**: Lines 400-600

**Purpose**: Decompose the bitmap into individual paths

**Key Functions**:
- `bm_to_pathlist()`: Main decomposition function
- `findnext()`: Find next black pixel
- `findpath()`: Trace a complete path around a region
- `xor_path()`: Remove traced regions from bitmap

**Process**:
1. Scan bitmap for black pixels
2. For each found pixel, trace a complete path around the region
3. Remove traced regions using XOR operation
4. Continue until no more black pixels remain

### Stage 2: Optimal Polygon Calculation
**Files**: Lines 800-1000

**Purpose**: Convert paths into optimal polygons

**Key Functions**:
- `_calc_sums()`: Precompute sums for fast calculations
- `_calc_lon()`: Find longest straight line segments
- `_bestpolygon()`: Calculate optimal polygon

**Process**:
1. Precompute sums for efficient calculations
2. Find longest straight line segments from each point
3. Calculate optimal polygon that approximates the path
4. Store polygon vertices

### Stage 3: Vertex Adjustment
**Files**: Lines 1000-1200

**Purpose**: Adjust polygon vertices for better curve fitting

**Key Functions**:
- `_adjust_vertices()`: Adjust vertices to minimize distance from original path

**Process**:
1. Calculate optimal line segments between polygon vertices
2. Find intersections of consecutive segments
3. Adjust vertices to minimize distance from original path
4. Ensure vertices stay within unit square

### Stage 4: Smoothing and Corner Analysis
**Files**: Lines 1200-1400

**Purpose**: Convert polygon into smooth curves

**Key Functions**:
- `_smooth()`: Convert polygon into smooth curves
- `reverse()`: Reverse curve orientation if needed

**Process**:
1. Analyze each vertex for corner vs curve classification
2. Calculate alpha values (smoothness parameters)
3. Convert straight segments to Bezier curves where appropriate
4. Mark sharp corners as corner segments

### Stage 5: Curve Optimization
**Files**: Lines 1400-1600

**Purpose**: Optimize curves by combining segments

**Key Functions**:
- `_opticurve()`: Optimize curve segments
- `opti_penalty()`: Calculate penalty for curve combinations

**Process**:
1. Analyze curve segments for potential combination
2. Calculate penalties for different combinations
3. Find optimal combination that minimizes total penalty
4. Replace multiple segments with single optimized curves

## Key Algorithms

### Path Tracing Algorithm
The path tracing algorithm uses a "left-hand rule" approach:
1. Start at a black pixel
2. Move along the boundary, always keeping black pixels to the left
3. Use turn policies to resolve ambiguous cases
4. Continue until returning to the starting point

### Turn Policies
The algorithm supports several turn policies for resolving ambiguous cases:
- `POTRACE_TURNPOLICY_BLACK`: Prefer black pixels
- `POTRACE_TURNPOLICY_WHITE`: Prefer white pixels
- `POTRACE_TURNPOLICY_LEFT`: Always turn left
- `POTRACE_TURNPOLICY_RIGHT`: Always turn right
- `POTRACE_TURNPOLICY_MINORITY`: Choose minority direction
- `POTRACE_TURNPOLICY_MAJORITY`: Choose majority direction
- `POTRACE_TURNPOLICY_RANDOM`: Use deterministic random choice

### Bezier Curve Fitting
The algorithm uses cubic Bezier curves for smooth curve representation:
- Control points are calculated based on alpha values
- Alpha values determine the smoothness of curves
- Curves are optimized to minimize deviation from original path

## Usage Examples

### Basic Usage
```python
import numpy as np
from potrace import Bitmap

# Create a simple bitmap
data = np.array([
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0]
], dtype=bool)

# Create bitmap and trace
bm = Bitmap(data)
path = bm.trace()

# Access curves
for curve in path.curves:
    for segment in curve.segments:
        if segment.is_corner:
            print(f"Corner at {segment.c}")
        else:
            print(f"Bezier: {segment.c1} -> {segment.c2} -> {segment.end_point}")
```

### Advanced Usage
```python
# Custom parameters
path = bm.trace(
    turdsize=2,           # Minimum area for paths
    turnpolicy=POTRACE_TURNPOLICY_MINORITY,
    alphamax=1.0,         # Maximum smoothness
    opticurve=True,       # Enable curve optimization
    opttolerance=0.2      # Optimization tolerance
)
```

## Mathematical Concepts

### Vector Operations
The implementation uses several mathematical concepts:
- **Cross Product**: Used for area calculations and orientation
- **Dot Product**: Used for distance calculations
- **Quadratic Forms**: Used for curve fitting and optimization

### Bezier Curves
Cubic Bezier curves are defined by four points:
- P0: Start point
- P1, P2: Control points
- P3: End point

The curve is calculated using the formula:
```
B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
```

### Optimization
The curve optimization uses:
- **Penalty Functions**: Measure deviation from original path
- **Dynamic Programming**: Find optimal curve combinations
- **Convexity Analysis**: Ensure curves maintain proper shape

## Performance Considerations

### Time Complexity
- **Path Decomposition**: O(n²) where n is the number of pixels
- **Polygon Calculation**: O(n²) for longest line calculation
- **Curve Optimization**: O(n³) in worst case, but typically much faster

### Memory Usage
- **Bitmap Storage**: O(width × height)
- **Path Storage**: O(number of paths × average path length)
- **Curve Storage**: O(number of segments)

### Optimization Tips
1. Use appropriate `turdsize` to filter out small artifacts
2. Adjust `alphamax` for desired smoothness
3. Use `opticurve=False` for faster processing if optimization isn't needed
4. Consider preprocessing images to reduce noise

## Limitations and Considerations

### Input Requirements
- Input must be a binary (black/white) image
- Image should be reasonably clean (low noise)
- Very complex images may produce many small paths

### Output Characteristics
- Output is always closed paths
- Curves are smooth but may not perfectly match original
- Sharp corners are preserved as corner segments

### Known Issues
- Tree structure for nested paths is not implemented
- Some advanced features from original Potrace are missing
- Memory usage can be high for large images

## Comparison with Original Potrace

This Python implementation follows the original Potrace algorithm closely but with some differences:
- **Language**: Python vs C
- **Data Structures**: Python objects vs C structs
- **Memory Management**: Automatic vs manual
- **Performance**: Generally slower due to Python overhead
- **Features**: Simplified version of original

## Future Improvements

Potential enhancements could include:
1. Implement tree structure for nested paths
2. Add support for color images
3. Optimize performance with Cython or numba
4. Add more output formats (SVG, PDF, etc.)
5. Implement parallel processing for large images

## References

This implementation is based on the original Potrace algorithm by Peter Selinger. For more information about the mathematical foundations and original implementation, see:
- [Potrace Homepage](http://potrace.sourceforge.net/)
- [Potrace Algorithm Documentation](http://potrace.sourceforge.net/potrace.pdf)

## License

This implementation follows the same license as the original Potrace project (GPL). 