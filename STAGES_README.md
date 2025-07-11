# Potrace Algorithm Stages Documentation

This document provides a detailed explanation of all 5 stages of the Potrace algorithm for converting bitmap images into smooth vector curves.

## Table of Contents

1. [Overview](#overview)
2. [Stage 1: Path Decomposition](#stage-1-path-decomposition)
3. [Stage 2: Optimal Polygon](#stage-2-optimal-polygon)
4. [Stage 3: Vertex Adjustment](#stage-3-vertex-adjustment)
5. [Stage 4: Smoothing and Corner Analysis](#stage-4-smoothing-and-corner-analysis)
6. [Stage 5: Curve Optimization](#stage-5-curve-optimization)
7. [Stage Interactions](#stage-interactions)
8. [Performance Characteristics](#performance-characteristics)

---

## Overview

The Potrace algorithm converts bitmap images into vector graphics through a sophisticated 5-stage pipeline. Each stage builds upon the results of the previous stage.

### Stage Flow

```
Bitmap Input → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Vector Output
```

---

## Stage 1: Path Decomposition

**Purpose**: Decompose bitmap into individual paths by tracing boundaries.

**Location**: `potrace/potrace.py` (lines 516-960)

### Key Functions

- **`bm_to_pathlist()`**: Main decomposition function
- **`findpath()`**: Core path tracing using "left-hand rule"
- **`findnext()`**: Find next black pixel to trace
- **`xor_path()`**: Remove traced regions from bitmap

### Algorithm

1. Find next black pixel using `findnext()`
2. Trace path boundary using `findpath()`
3. Remove traced region using `xor_path()`
4. Repeat until no black pixels remain

### Turn Policies

- `POTRACE_TURNPOLICY_BLACK`: Prefer black pixels
- `POTRACE_TURNPOLICY_WHITE`: Prefer white pixels
- `POTRACE_TURNPOLICY_LEFT`: Always turn left
- `POTRACE_TURNPOLICY_RIGHT`: Always turn right
- `POTRACE_TURNPOLICY_MINORITY`: Choose less common direction
- `POTRACE_TURNPOLICY_MAJORITY`: Choose more common direction
- `POTRACE_TURNPOLICY_RANDOM`: Use deterministic random choice

### Output

Produces list of `_Path` objects with:
- `pt`: Boundary points
- `area`: Enclosed area
- `sign`: Path orientation (positive/negative)

---

## Stage 2: Optimal Polygon

**Purpose**: Find optimal polygon with minimum vertices.

**Location**: `potrace/potrace.py` (lines 1471-1750)

### Key Functions

- **`_calc_sums()`**: Precompute cumulative sums
- **`_calc_lon()`**: Calculate longest straight lines
- **`_bestpolygon()`**: Find optimal polygon using dynamic programming
- **`penalty3()`**: Calculate edge penalty

### Algorithm

1. Calculate cumulative sums for fast computations
2. Find longest straight lines from each point
3. Use dynamic programming to find optimal polygon
4. Minimize total penalty while maintaining accuracy

### Dynamic Programming

- `pen[j]`: Minimum penalty to reach point j
- `prev[j]`: Previous point in optimal path
- `len[j]`: Number of segments in optimal path

### Output

Produces:
- `_m`: Number of vertices in optimal polygon
- `_po`: Array of vertex indices in optimal order

---

## Stage 3: Vertex Adjustment

**Purpose**: Adjust vertices to create smooth curves.

**Location**: `potrace/potrace.py` (lines 1751-1850)

### Key Functions

- **`_adjust_vertices()`**: Adjust vertices for smooth curves
- **`pointslope()`**: Calculate center and direction of line segments

### Algorithm

1. Calculate "optimal" point-slope representation for each line
2. Represent lines as quadratic forms
3. Find intersection points that minimize distance
4. Constrain points to unit square for numerical stability

### Quadratic Form Representation

Each line segment is represented as quadratic form Q where distance from point (x,y) to line is (x,y,1)Q(x,y,1)^t.

### Output

Produces `_curve` array with adjusted vertices optimized for smooth curve generation.

---

## Stage 4: Smoothing and Corner Analysis

**Purpose**: Determine sharp corners vs smooth curves.

**Location**: `potrace/potrace.py` (lines 1851-2020)

### Key Functions

- **`_smooth()`**: Analyze vertices and classify as corners or curves
- **`reverse()`**: Reverse orientation of negative paths

### Smoothness Parameter (Alpha)

```python
dd = dpara(curve[i].vertex, curve[j].vertex, curve[k].vertex) / denom
alpha = (1 - 1.0 / dd) if dd > 1 else 0
alpha = alpha / 0.75
```

### Classification Logic

- **High Alpha (≥ alphamax)**: Sharp corner → `CornerSegment`
- **Low Alpha (< alphamax)**: Smooth curve → `BezierSegment`

### Bezier Control Points

For smooth curves, control points calculated using alpha:
```python
p2 = interval(0.5 + 0.5 * alpha, curve[i].vertex, curve[j].vertex)
p3 = interval(0.5 + 0.5 * alpha, curve[k].vertex, curve[j].vertex)
```

### Output

Produces `_curve` with segment types assigned:
- `CornerSegment` objects for sharp corners
- `BezierSegment` objects for smooth curves

---

## Stage 5: Curve Optimization

**Purpose**: Combine multiple Bezier segments into single segments.

**Location**: `potrace/potrace.py` (lines 2021-2400)

### Key Functions

- **`_opticurve()`**: Optimize curve segments using dynamic programming
- **`opti_penalty()`**: Calculate penalty for curve optimization

### Algorithm

1. Pre-calculate convexity and area data
2. Use dynamic programming to find optimal path
3. Reconstruct optimized curve with fewer segments

### Optimization Constraints

1. **Convexity**: Preserve curve convexity
2. **Corner-freeness**: Avoid sharp corners in optimized segments
3. **Bend Limit**: Maximum bend angle < 179 degrees
4. **Tangency**: Maintain tangency with original curve edges
5. **Corner Compatibility**: Respect original corner constraints

### Penalty Calculation

Includes:
- **Edge Tangency**: Distance from optimized curve to original edges
- **Corner Compatibility**: Distance from optimized curve to original corners
- **Smoothness**: Encourages smooth transitions

### Output

Produces:
- `_ocurve`: Optimized curve with fewer segments
- `s[]`, `t[]`: Curve parameters for smooth transitions
- `beta[]`: Transition parameters between segments

---

## Stage Interactions

### Data Flow

```
Stage 1: _Path.pt (boundary points) → Stage 2
Stage 2: _Path._po (polygon vertices) → Stage 3
Stage 3: _Path._curve (adjusted vertices) → Stage 4
Stage 4: _Path._curve (segment types) → Stage 5
Stage 5: _Path._ocurve (optimized curves) → Final Output
```

### Parameter Dependencies

- **Stage 1**: `turdsize`, `turnpolicy`
- **Stage 2**: No additional parameters
- **Stage 3**: No additional parameters
- **Stage 4**: `alphamax`
- **Stage 5**: `opttolerance`

---

## Performance Characteristics

### Time Complexity

| Stage | Complexity | Description |
|-------|------------|-------------|
| 1 | O(n²) | Path decomposition with optimized longest line calculation |
| 2 | O(n²) | Dynamic programming for optimal polygon |
| 3 | O(m) | Vertex adjustment where m = polygon vertices |
| 4 | O(m) | Smoothing and corner analysis |
| 5 | O(m²) | Curve optimization with dynamic programming |

### Memory Usage

- **Stage 1**: Creates _Path objects for each region
- **Stage 2**: Precomputes sums and longest lines
- **Stage 3**: Creates _Curve objects with adjusted vertices
- **Stage 4**: Assigns segment types
- **Stage 5**: Creates optimized curves

### Optimization Tips

1. **Stage 1**: Use `turdsize` to filter small paths
2. **Stage 2**: Optimized longest line calculation (31x speedup)
3. **Stage 4**: Adjust `alphamax` for corner vs curve balance
4. **Stage 5**: Use `opticurve=False` to skip optimization

### Overall Complexity

- **Best Case**: O(n²) for simple shapes
- **Worst Case**: O(n³) for complex shapes
- **Average Case**: O(n²) for typical images

---

This documentation provides a comprehensive overview of all 5 stages of the Potrace algorithm, their purposes, implementations, and interactions. Each stage builds upon the previous one to transform raw bitmap data into optimized vector curves. 