# Potrace Algorithm Stages

The Potrace algorithm transforms bitmap images into smooth vector curves through a sophisticated five-stage pipeline. Each stage builds upon the previous one, progressively converting raw pixel boundaries into elegant mathematical curves. The algorithm's power lies in its ability to preserve the essential character of the original image while eliminating the jagged edges and pixel artifacts inherent in bitmap representations.

## Stage 1: Path Decomposition

The journey begins with Stage 1, where the algorithm decomposes the bitmap into individual paths by tracing the boundaries of black regions. This stage employs a clever boundary-following algorithm that starts at the upper-left corner of each black region and traces the complete perimeter using the "left-hand rule." The algorithm maintains a direction vector and systematically explores the neighborhood around each pixel to determine the next step along the boundary. When the algorithm encounters ambiguous situations where multiple directions are equally valid, it applies a turn policy to resolve the ambiguity. These turn policies can prefer black pixels, white pixels, left turns, right turns, or use deterministic random choices based on the local geometry.

The path tracing process continues until the algorithm returns to the starting point, completing a closed loop. Each traced path is stored as a sequence of points representing the boundary, along with metadata such as the enclosed area and whether it represents a positive or negative region. The algorithm also includes a "turdsize" parameter that filters out paths below a certain area threshold, eliminating noise and small artifacts that would not contribute meaningfully to the final result. This stage produces a list of path objects, each containing the raw boundary points and associated properties that will be processed in subsequent stages.

## Stage 2: Optimal Polygon Calculation

Stage 2 transforms the raw boundary paths into optimal polygons with the minimum number of vertices while maintaining accuracy. This stage uses the longest line information computed in Stage 1 to determine which points can be connected by straight lines. The algorithm employs dynamic programming to find the optimal polygon that minimizes the number of vertices while staying within acceptable error bounds.

The process begins by calculating "longest lines" from each point - the furthest point that can be reached by a straight line while maintaining accuracy. These longest lines are determined through an efficient constraint-based algorithm that only tests corner points rather than every possible combination, making it significantly faster than the original O(n³) approach. The algorithm uses geometric constraints to ensure that the straight lines accurately represent the original path, and it handles the cyclic nature of closed paths through careful index arithmetic.

Once the longest lines are computed, the algorithm uses dynamic programming to find the optimal polygon. It considers all possible combinations of vertices and selects the one that minimizes the total number of vertices while keeping the approximation error below acceptable thresholds. The result is a polygon with significantly fewer vertices than the original path, but one that still accurately represents the shape. This polygon serves as the foundation for the curve fitting that occurs in later stages.

## Stage 3: Vertex Adjustment

Stage 3 takes the optimal polygon from Stage 2 and adjusts the vertices to create smooth curves. This stage calculates the intersection of adjacent line segments and ensures the resulting points are positioned optimally for curve fitting. The algorithm represents each line segment as a quadratic form, which allows it to calculate the optimal intersection points mathematically.

The vertex adjustment process involves solving systems of linear equations to find the best intersection points. When the equations are well-conditioned, the algorithm can solve them directly. However, when the lines are nearly parallel or the system is singular, the algorithm adds orthogonal constraints to ensure numerical stability. The algorithm also includes boundary checking to ensure that the adjusted vertices remain within reasonable bounds, preventing numerical instabilities that could affect the quality of the final curves.

The adjusted vertices are stored in the curve structure along with their associated line segments. These adjusted vertices provide the foundation for the smooth curve generation that occurs in Stage 4. The adjustment process is crucial because it ensures that the vertices are positioned optimally for creating smooth Bezier curves, rather than simply using the original polygon vertices which might not be ideal for curve fitting.

## Stage 4: Smoothing and Corner Analysis

Stage 4 determines which vertices should become sharp corners and which should be smoothed into curves. This stage analyzes the local geometry around each vertex to calculate a smoothness parameter called alpha. The alpha value ranges from 0 to 1, where higher values indicate sharper corners and lower values indicate smoother curves.

The algorithm examines each vertex and calculates how well it can be approximated by a smooth curve versus a sharp corner. It considers the angles between adjacent line segments and the local curvature to determine the optimal smoothness. Vertices with high alpha values become sharp corners, while those with low alpha values become smooth curve segments. The algorithm also includes bounds checking to ensure that alpha values remain within reasonable ranges, preventing extreme values that could lead to poor curve quality.

For vertices that become smooth curves, the algorithm calculates control points for cubic Bezier curves. These control points are positioned to create smooth transitions between adjacent curve segments while maintaining accuracy to the original path. The algorithm uses geometric calculations to ensure that the Bezier curves accurately represent the local geometry and provide smooth transitions between segments.

The smoothing process also handles the orientation of paths. Negative paths, which represent white regions or holes, have their orientation reversed to ensure consistent curve generation. This ensures that all curves follow a consistent winding order, which is important for rendering and further processing.

## Stage 5: Curve Optimization

The final stage optimizes the curves by combining multiple Bezier segments into single segments where possible. This stage uses dynamic programming to find the optimal combination of curve segments that minimizes the total penalty while maintaining accuracy within specified tolerance bounds.

The optimization process begins by pre-calculating convexity information for each curve segment. This information helps the algorithm determine which segments can be combined while maintaining the overall shape characteristics. The algorithm also pre-calculates area information to enable fast computation of curve properties during optimization.

The core of the optimization is the penalty calculation, which measures how well a combined curve segment approximates the original path. The penalty considers factors such as tangency with the original path, corner preservation, and maximum bend angles. The algorithm ensures that combined curves maintain the essential characteristics of the original path while reducing the total number of segments.

The optimization process uses dynamic programming to find the globally optimal combination of curve segments. It considers all possible ways to combine adjacent segments and selects the combination that minimizes the total penalty while staying within the specified tolerance bounds. The result is a set of optimized curves that accurately represent the original bitmap with fewer segments than the initial curve generation.

The final output of the Potrace algorithm is a collection of smooth vector curves that accurately represent the original bitmap image. These curves can be rendered at any resolution without loss of quality, making them ideal for applications such as logo design, illustration, and any situation where scalable graphics are required. The algorithm's ability to preserve the essential character of the original image while eliminating pixel artifacts makes it a powerful tool for bitmap-to-vector conversion. 