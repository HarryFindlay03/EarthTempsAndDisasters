import math

def distance_between_points(point_a, point_b):
    """
    Using the haversine distance formulate to calculate the distance between two lat long points

    Args:
        point_a (tuple): (lat, long) of point A
        point_b (tuple): (lat, long) of point B

    Returns:
        float: The distance between point A and point B
    """
    theta_1, theta_2 = point_a[0], point_b[0]
    phi_1, phi_2 = point_a[1], point_b[1]

    R = 6371

    dLat = deg2Rad(theta_2 - theta_1)
    dLong = deg2Rad(phi_2 - phi_1)

    a = math.sin(dLat / 2) * math.sin(dLat/2) + math.cos(deg2Rad(theta_1)) * math.cos(deg2Rad(theta_2)) * math.sin(dLong/2) * math.sin(dLong/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c

    return d

def deg2Rad(degrees):
    return degrees * (math.pi / 180)