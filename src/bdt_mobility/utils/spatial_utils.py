import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, LineString, Point
import math


def buffer_around(
    gdf : gpd.GeoDataFrame,
    buffer_size : float = 3000.0
):
    """
    Creates a buffer around the geometries in the provided GeoDataFrame.

    Args:
        `gdf` (`gpd.GeoDataFrame`): GeoDataFrame containing geometries to buffer.
        `buffer_size` (`float`): Size of the buffer in map units (e.g., meters). Defaults to `3000.0`.

    Returns:
        : A GeoSeries with buffered geometries.
    """
    return gdf.buffer(buffer_size)


def equivalent_circle(
    polygon
):
    """
    Given a geolocated Polygon, returns a circle with the same centroid and the same area.

    Args:
        `polygon` XXX

    Returns:
        XXXX
    """
    centroid = polygon.centroid
    x = centroid.coords[0][0]
    y = centroid.coords[0][1]
    radius = (polygon.area / 3.14159) ** 0.5  # Radius from area of circle: A = πr²
    return x, y, centroid.buffer(radius), radius


def boundary_points(
    geom, 
    n: int = 8
):
    """
    Given a geolocated Polygon, returns the n equally-spaced Points in its exterior border

    Args:
        `geom`: The Polygon geometry.
        `n` (`int`): The number of Points to be sampled. Default to 8.

    Returns:
        XXXXXX
    """
    return [geom.exterior.interpolate(d) for d in np.linspace(0, geom.exterior.length, n, endpoint=False)]


def closest_point(
    polygon: Polygon, 
    points_gdf: gpd.GeoDataFrame
):
    """
    Finds the ID of the closest point in a GeoDataFrame to a given polygon.    
    Calculates the distance from each point in the GeoDataFrame to the input polygon
    and returns the ID of the point with the minimum distance.
    
    Args:
        polygon (shapely.geometry.Polygon): The polygon geometry to measure distances from.
        points_gdf (gpd.GeoDataFrame): GeoDataFrame containing point geometries and an 'id' column for identifiers.
        
    Returns:
        The ID of the closest point to the polygon.
    """
    distances = points_gdf.distance(polygon)
    min_distance_index = distances.idxmin()
    closest = points_gdf.iloc[min_distance_index]['id']
    return closest


def unique_nearest_lines(
    pts : gpd.GeoDataFrame, 
    lns : gpd.GeoDataFrame
) -> pd.DataFrame:  
    """
    Finds the closest linestring(s) in a GeoDataFrame for each point in another GeoDataFrame.    
    For each point, identifies all linestrings at the minimum distance and creates paired records
    containing both the point data and the matching linestring data.
    
    Args:
        `pts` (`gpd.GeoDataFrame`): GeoDataFrame containing point geometries.
        `lns` (`gpd.GeoDataFrame`): GeoDataFrame containing linestring geometries
                                           with 'u', 'v', and 'key' columns.
    
    Returns:
        `pd.DataFrame`: DataFrame containing combined data from points and their
                      matching nearest linestrings. Each row represents a point-linestring pair,
                      with multiple rows possible for a single point if multiple linestrings
                      are at the minimum distance.
    """  
    full_closest_linestrings = []
    for _, point in pts.iterrows():
        distances = lns.geometry.distance(point.geometry)
        min_distance = distances.min()
        min_distance_indices = distances[distances == min_distance].index.tolist()

        for closest_index in min_distance_indices:
            closest_linestring = lns[['u','v','key']].loc[closest_index]
            combined_row = pd.concat([point, closest_linestring])
            combined_row["distance_to_road"] = min_distance
            full_closest_linestrings.append(combined_row)

    return pd.DataFrame(full_closest_linestrings)


def nearest_lines_within_buffer(
    pts : gpd.GeoDataFrame, 
    lns : gpd.GeoDataFrame,
    max_search_distance : int|None = None,
) -> pd.DataFrame:  
    """
    Finds the closest linestring(s) in a GeoDataFrame for each point in another GeoDataFrame.    
    For each point, identifies all linestrings at the minimum distance and creates paired records
    containing both the point data and the matching linestring data.
    
    Args:
        `pts` (`gpd.GeoDataFrame`): GeoDataFrame containing point geometries.
        `lns` (`gpd.GeoDataFrame`): GeoDataFrame containing linestring geometries
                                           with 'u', 'v', and 'key' columns.
        `max_search_distance` (`int|None`): max distance in meters where to search for the lines (default: `None`)
    
    Returns:
        `pd.DataFrame`: DataFrame containing combined data from points and their
                      matching nearest linestrings. Each row represents a point-linestring pair,
                      with multiple rows possible for a single point if multiple linestrings
                      are at the minimum distance.
    """  
    full_closest_linestrings = []
    for idx, point in pts.iterrows():
        distances = lns.geometry.distance(point.geometry).sort_values()
        if max_search_distance is None:
            min_distance_indices = distances.index.tolist()
            min_distance = distances.to_list()
        else:
            min_distance_indices = distances[distances<max_search_distance].index.tolist()
            min_distance = distances[distances<max_search_distance].to_list()

        for idx, closest_index in enumerate(min_distance_indices):
            closest_linestring = lns[['u','v','key']].loc[closest_index]
            combined_row = pd.concat([point, closest_linestring])
            combined_row["distance_to_road"] = min_distance[idx]
            full_closest_linestrings.append(combined_row)

    return pd.DataFrame(full_closest_linestrings)


def is_left(
    linestring: LineString, 
    point: Point
) -> bool:    
    """
    Determines if a point is to the left of the closest segment of a linestring.

    This function finds the closest segment of the linestring to the given point,
    then uses coordinate translation and rotation to determine if the point lies
    to the left of that segment when looking from the first vertex toward the second.

    The algorithm works by:
    1. Finding the closest segment of the linestring to the point
    2. Translating the coordinate system to make the first vertex the origin
    3. Rotating the coordinate system to align the segment with the x-axis
    4. Checking if the transformed point's y-coordinate is positive (left side)

    Args:
        linestring (LineString): A shapely LineString object to test against.
        point (Point): A shapely Point object to test.
        
    Returns:
        bool: True if the point is to the left of the closest segment,
                False if the point is to the right or on the segment.
                None if no closest segment is found (should not occur).
    """
    def translate_point(p, p0):
        p[0] = p[0] - p0[0]
        p[1] = p[1] - p0[1]
        return p
    def rotate_point(p, angle):
        p = [p[0]*math.cos(angle)-p[1]*math.sin(angle),
          p[0]*math.sin(angle)+p[1]*math.cos(angle) ]
        return p
    def convert_angle (angle, yy):
        if yy < 0:
            angle = - angle
        return angle

    dist = float('inf')
    closest_segment_index = -1
    for i in range(len(linestring.coords) - 1):
        segment = LineString([linestring.coords[i], linestring.coords[i + 1]])
        dist_temp = point.distance(segment)
        if dist_temp < dist:
            dist = dist_temp
            closest_segment_index = i
    if closest_segment_index == -1:
        return None  # This should not happen

    p1 = linestring.coords[closest_segment_index]
    p2 = linestring.coords[closest_segment_index + 1]

    p2 = translate_point([p2[0],p2[1]], [p1[0],p1[1]])
    pp = translate_point([point.x,point.y],  [p1[0],p1[1]])

    angle_p1p2 = math.acos(p2[0] / math.sqrt(p2[1]**2 + p2[0]**2))
    angle_p1p2 = convert_angle(angle_p1p2, yy=p2[1])      

    angle_p1p2 = -angle_p1p2
    p2 = rotate_point(p2, angle_p1p2)
    pp = rotate_point(pp, angle_p1p2)
    
    angle_p1point = math.acos((pp[0]) / math.sqrt(pp[1]**2 + pp[0]**2))
    angle_p1point = convert_angle(angle_p1point, yy=pp[1])

    left_condition = angle_p1point > 0
    return left_condition  # True if the point is to the left of the segment, False otherwise


def direction_wrt_polygon(
    line: gpd.GeoDataFrame,
    polygon: Polygon
) -> str:
    """
    Determines the direction of a line with respect to a polygon.
    
    This function evaluates whether a line is 'incoming' to or 'outgoing' from a polygon
    based on the positions of its start and end points relative to the polygon.
    
    Args:
        row (gpd.GeoDataFrame) : A row from a GeoDataFrame containing a LineString geometry.
        polygon (shapely.Polygon) : A polygon geometry object to check against.
        
    Returns:
        str : 'incoming' if the line is moving toward the polygon, or 'outgoing' if the line is moving away from the polygon.
    """
    start_point = line.coords[0]
    end_point = line.coords[-1]
    
    start_point_geom = Point(start_point)
    end_point_geom = Point(end_point)
    
    start_inside = polygon.contains(start_point_geom)
    end_inside = polygon.contains(end_point_geom)
    
    if start_inside and not end_inside:
        return 'outgoing'
    elif not start_inside and end_inside:
        return 'incoming'
    elif start_inside and end_inside:
        distance_start = start_point_geom.distance(polygon.exterior)
        distance_end = end_point_geom.distance(polygon.exterior)
        return 'incoming' if distance_start < distance_end else 'outgoing'
    else:
        distance_start = start_point_geom.distance(polygon)
        distance_end = end_point_geom.distance(polygon)
        return 'outgoing' if distance_start < distance_end else 'incoming'
    

def calculate_direction(line):
    start, end = line.coords[0], line.coords[-1]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = np.sqrt(dx**2 + dy**2)
    return (dx / length, dy / length)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2)


def alignment_angle(
    point_obj: Point | gpd.GeoSeries, 
    line_obj: LineString | gpd.GeoSeries, 
    compass_angle: float
) -> float:
    """
    Compute the signed angular difference between a compass bearing and the
    orientation of the nearest segment of a line geometry.

    The function projects a point onto a LineString, identifies the line segment
    that contains the projection, and returns the difference between the provided
    compass angle and the segment's bearing.  A positive result means the compass
    direction is clockwise of the segment direction; negative means counter-clockwise.

    Parameters
    ----------
    point_obj : shapely.geometry.Point or geopandas.GeoSeries
        The point of interest. If a GeoSeries is passed, only the first element is used.
    line_obj : shapely.geometry.LineString or geopandas.GeoSeries
        The reference line. If a GeoSeries is passed, only the first element is used.
    compass_angle : float
        The bearing to evaluate, expressed in compass degrees
        (North = 0°, clockwise positive, range 0–360).

    Returns
    -------
    float
        The signed angular difference in degrees, in the range [-180, 180].
        A value of 0 means the bearing is perfectly aligned with (parallel to)
        the nearest segment; ±90 means perpendicular; ±180 means anti-parallel.

    Notes
    -----
    - Distances and projections are computed in the coordinate reference system
      of the input geometries; make sure both objects share the same CRS and
      that it is metric if you need the ``distance_to_line`` value (not returned
      but computed internally) to be in metres.
    - A small epsilon (1e-7) is used when matching the projected distance to a
      segment in order to avoid floating-point boundary misses.
    - If the projection falls exactly at the very end of the line and no segment
      match is found, the last segment is used as a fallback and the returned
      value is folded into [0, 90] (unsigned acute angle).
    """
    p = point_obj.iloc[0] if hasattr(point_obj, "iloc") else point_obj
    l = line_obj.iloc[0] if hasattr(line_obj, "iloc") else line_obj

    # 1. Physical distance from point to the line (Meters)
    distance_to_line = l.distance(p)

    # 2. Mile marker: how far along the line is the projection?
    projected_dist = l.project(p)

    # 3. Compass (N=0, CW) -> Cartesian (E=0, CCW)
    point_math_angle = (90 - compass_angle) % 360

    coords = list(l.coords)
    final_angle_diff = None
    
    # We loop through segments to find the match
    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i+1]
        
        # Get mile markers for this segment
        d1 = l.project(Point(p1))
        d2 = l.project(Point(p2))
        
        # We add a tiny epsilon (1e-7) to handle the 'None' errors 
        # caused by floating point precision
        if d1 <= projected_dist + 1e-7 and projected_dist - 1e-7 <= d2:
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            seg_math_angle = math.degrees(math.atan2(dy, dx))
            
            # Difference calculation
            diff = (point_math_angle - seg_math_angle + 180) % 360 - 180
            
            final_angle_diff = diff
            break

    # 4. Fallback: If None (due to being exactly at the tip), use the last segment
    if final_angle_diff is None:
        # Just use the very last segment of the line
        p1, p2 = coords[-2], coords[-1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        seg_math_angle = math.degrees(math.atan2(dy, dx))
        diff = (point_math_angle - seg_math_angle + 180) % 360 - 180
        final_angle_diff = abs(diff) if abs(diff) <= 90 else 180 - abs(diff)
            
    return final_angle_diff # [-180;180]