def get_centre_of_bbox(bbox):
    """
    Get the centre of the bounding box
    """
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    """
    Get the width of the bounding box
    """
    return bbox[2]-bbox[0]

def measure_distance(p1,p2):
    """
    Measure the distance between two points
    """
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(bbox):
    """
    Get the foot position of the player
    """
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)