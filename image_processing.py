import matplotlib.pyplot as plt
import imageio as img
import numpy as np
from skimage import draw, feature, filters, morphology, transform, util
from geometry import *

def find_all(file_name, num_sources=10, num_oas=5, printed=False):
    """Main function of this file. Identifies all structures in an image and returns them
    data structure: TODO """
    image = bw_image(file_name, printed=printed)
    circles = circle_hough(image, num_sources)
    lines = line_hough(image, printed=printed)
    original_lines = list(lines)
    classify_circles(image, lines, circles, printed=printed)
    filtered_lines = consolidate_lines(lines, image, circles)
    triangles = process_triangles(image, filtered_lines, num_oas, printed=printed)
    resistors = process_resistors(image, filtered_lines, printed=printed)
    drawn_image = draw_shapes([filtered_lines, circles, triangles], file_name)
    show(drawn_image)
    return Image(image, filtered_lines, circles, triangles)


### MISCELLANEOUS HELPER FUNCTIONS ###

def bw_image(file_name, printed=False):
    """
    Returns a black and white, contrastified, eroded, smoothed image.

    file_name: string file name of an image file to be read."""
    processed = img.imread(file_name, as_gray=True).astype('uint8')
    if not printed:
        processed = morphology.erosion(filters.median(processed))
    x_size, y_size = processed.shape
    # Split up to deal with shadows and changes in light
    # Order goes top left -> top right -> bottom left -> bottom right
    quads = [processed[0:x_size//2, 0:y_size//2], processed[x_size//2:x_size, 0:y_size//2], processed[0:x_size//2, y_size//2:y_size], processed[x_size//2:x_size, y_size//2:y_size]]
    if printed:
        midpoint_percent = 0.85
    else:
        midpoint_percent = 0.55
    quad1, quad2, quad3, quad4 = [contrastify(quad, midpoint_percent) for quad in quads]
    final_img = np.vstack((np.hstack((quad1, quad3)), np.hstack((quad2, quad4))))
    return final_img

def contrastify(arr, midpoint_percent=0.55):
    """
    Takes a numpy array and returns a new one separated at a midpoint, mapping smaller values to zero and larger ones to a maximum value.

    image: numpy array of numbers
    midpoint: lower values are mapped to zero, equal or higher values are mapped to max_val
    """
    midpoint_thresh = np.min(arr) * midpoint_percent + np.max(arr) * midpoint_percent
    arr = (arr > midpoint_thresh) * 255
    return arr

def draw_shapes(list_of_shapes, file_name):
    """ Draws all given shapes onto an image.
    list_of_shapes is a list of shape lists (e.g. a [[all circles], [all lines]]), whose members have a draw function"""
    color_image = plt.imread(file_name)
    color_image.setflags(write=1)
    # Uncomment this to display detected shapes without the original image
    # color_image = np.zeros(color_image.shape)
    for shape_type in list_of_shapes:
        for shape in shape_type:
            shape.draw(color_image)
    return color_image

def show_bw(image):
    plt.imshow(image, cmap=plt.cm.gray, interpolation='gaussian')
    plt.show()

def show(image):
    plt.imshow(image)
    plt.show()

def show_plot(file_name):
    """
    Contrastifies an image and then displays it through matplotlib.pyplot

    file_name: string name of an image file to be read
    """
    image = bw_image(file_name)
    show(contrastify(edgify(image)))

### ABSTRACTING FUNCTIONS ###
# These are here so that the find_all function is readable


def classify_circles(image, lines, circles, printed=False):
    for circ in circles:
        circ.classify(image, lines, printed=printed)

def process_triangles(image, filtered_lines, num_oas, printed=False):
    """ Detects and classifies all triangles in an image; returns a list of triangles """
    if printed:
        mll, d_thresh = 0.05 * np.mean(image.shape), 0.03 * np.mean(image.shape)
    else:
        mll, d_thresh = 0.1 * np.mean(image.shape), 0.06 * np.mean(image.shape) 
    triangles = find_triangles(filtered_lines, min_line_length=mll, dist_threshold=d_thresh, max_triangles=num_oas)
    if not triangles:
        print('no triangles found :(')
    else:
        for triangle in triangles:
            triangle.classify(image, filtered_lines, printed=printed)
            print(triangle.orientation)
            print()
    return triangles

def process_resistors(image, filtered_lines, printed=False):
    """ Given processed lines, finds resistors and classifies their orientation.
    NOTE: As of now resistors can either be vertical or horizontal.
    TODO: Add functionality for diagonal resistors (Wheatstone bridge)"""
    def closest_clusters(ref_line, clusters):
        """ Returns the clusters (list) of lines which have at least one line sufficiently close to ref_line """
        dist_func = lambda cluster: min_cluster_distance(ref_line, cluster) < dist_thresh
        return list(filter(dist_func, clusters))
        return min(clusters, key=lambda cluster: min_cluster_distance(ref_line, cluster), default=[])

    def min_cluster_distance(ref_line, cluster):
        """ Returns the smallest distance from any line in the cluster to the reference line """
        dist_func = lambda line: ref_line.min_endpoint_dist(line)
        return min(map(dist_func, cluster), default=float('inf'))

    dist_thresh = np.mean(image.shape) * 0.1
    print(dist_thresh)
    diag_lines = [line for line in filtered_lines if line.angle_type in ('pos-diag', 'neg-diag')]
    clusters = []
    for line in diag_lines:
        # Close
        closest = closest_clusters(line, clusters)
        if not closest:
            clusters.append([line])
        elif len(closest) == 1:
            closest[0].append(line)
        else:
            big_cluster = []
            for cluster in closest:
                clusters.remove(cluster)
                big_cluster.extend(cluster)
            clusters.append(big_cluster)
    for cluster in clusters:
        for line in cluster:
            print(line)
        print()
            

### K-MEANS SECTION ###


### LINE SECTION ###

def find_lines(file_name, show_lines=False):
    """Performs the full process of identifying lines in an image."""
    return find_shape(file_name, 'line', show_shapes=show_lines)

def line_hough(image, printed=False):
        
    # Find edges
    edges = feature.canny(image, sigma=0.5, low_threshold=0.1, high_threshold=0.2, use_quantiles=True)
    if printed:
        line_length, line_gap = int(0.01 * np.mean(image.shape)), int(0.0025 * np.mean(image.shape))
    else:
        line_length, line_gap = int(0.02 * np.mean(image.shape)), int(0.0025 * np.mean(image.shape))
    line_coords = transform.probabilistic_hough_line(edges, threshold=0, line_length=line_length, line_gap=line_gap) 
    lines = [Line(c0[0], c0[1], c1[0], c1[1]) for c0, c1 in line_coords if not is_edge_line(c0, c1, image)]
    return lines

def is_edge_line(c0, c1, image, threshold=5):
    """ Returns whether the two coordinates are close enough to the edge to be eliminated. """
    return bool(min(c0 + c1) < threshold or max(c0[0], c1[0]) > image.shape[1] - threshold or max(c0[1], c1[1]) > image.shape[0] - threshold)

def consolidate_lines(lines, image, circles=[]):
    """ Takes all of the shorter lines returned by the line_hough function and combines them into longer lines.

    Input: 
        lines: a list of line objects
        circles: a list of circle objects
    Output: a list of lines of the same format
    
    Strategy: take the first line and find the next closest line.
              If this line is closer than some threshold, then join the two
                
              If the lines are of the same angle type, then combine them into one longer wire
                Then remove the old wires from the list and call the function on the list including the new wire
    """
    def consolidate_helper(list_of_lines):
        """ Recursively traverses the list of lines and joins together those which are close together.
        Returns the final list of lines.

        TODO: Add functionality for vertical and diagonal lines.
        (Possibly) add an "absolute minimum distance" function using projections
        
        """
        if len(list_of_lines) <= 1:
            return list_of_lines
        first_line = list_of_lines.pop(0)
        closest_line = find_closest_line(list_of_lines, first_line)
        if closest_line:
            joined_line = first_line.join_lines(closest_line)
            list_of_lines.remove(closest_line)
            list_of_lines.insert(0, joined_line)
            return consolidate_helper(list_of_lines)
        else:
            return [first_line] + consolidate_helper(list_of_lines)

    def find_closest_line(list_of_lines, ref_line):
        """ Finds the closest line to a given line of the same type. If there is no close line, return None. """
        if ref_line.angle_type in ['horz', 'vert']:
            look = rect(ref_line, width, width, dist)
        else:
            look = trap(ref_line, width, dist)
        closest = min([line for line in list_of_lines if line.angle_type == ref_line.angle_type and (look(line.coords[0]) or look(line.coords[1]))], key=lambda lin: ref_line.min_endpoint_dist(lin), default=None)
        return closest


    # Actual body of the function
    dist, width, min_proportion_filled = int(0.02 * np.mean(image.shape)), int(0.03 * np.mean(image.shape)), 0.2
    filtered_lines = consolidate_helper(lines)
    return filtered_lines

def rect(ref_line, ld_width, ru_width, dist, disp=False):
    """ Returns a rectangular look function based on the reference line.
    lu_width is the width in the left or upwards direction
    rd_width is the width in the right or downwards direction
    These variables used later on in triangle classification.
    """
    if ref_line.angle_type == 'horz':
        # Rectangle that is width wide from the center of the line in the direction of the first endpoint
        right_x, left_x= max(ref_line.x0, ref_line.x1), min(ref_line.x0, ref_line.x1)
        # look is a function that takes coordinates and returns whether they are in the rectangle
        bl_corner = (left_x - dist, ref_line.center_coord[1] - ld_width)
        tr_corner = (right_x + dist, ref_line.center_coord[1] + ru_width)
    elif ref_line.angle_type == 'vert':
        top_y, bottom_y  = max(ref_line.y0, ref_line.y1), min(ref_line.y0, ref_line.y1)
        bl_corner = (ref_line.center_coord[0] - ld_width, bottom_y - dist)
        tr_corner = (ref_line.center_coord[0] + ru_width, top_y + dist)
    def in_rect(coord):
        """ Determines if coord is within a rectangle spanned by 2 given corners. All coordinates are given in tuples. """
        return bl_corner[0] < coord[0] < tr_corner[0] and bl_corner[1] < coord[1] < tr_corner[1]
    # This is just here for debugging
    if disp:
        print('BL corner ' + str(bl_corner))
        print('TR corner ' + str(tr_corner))
        print()
    return in_rect

def trap(ref_line, width, dist):
    """ Returns a trapezoidal look function based on the reference line.
    NOTE: higher y values are lower, due to the nature of numpy arrays.
    As such, bottom refers to the LOWER of the two points and so on"""

    # Possible to scale width down for diagonal lines
    diag_width = width
    bottom, top = max(ref_line.coords, key=lambda x: x[1]), min(ref_line.coords, key=lambda x: x[1])
    slopes_right = ref_line.slope < 0
    extension = dist
    if slopes_right:
        left_corner = (bottom[0] - diag_width - extension, bottom[1] + extension)
        right_corner = (top[0] + diag_width + extension, top[1] - extension)
    else:
        left_corner = (top[0] - diag_width - extension, top[1] + extension)
        right_corner = (bottom[0] + diag_width + extension, bottom[1] - extension)

    # These line functions based on y = m(x - x0) + y0
    left_line = lambda x: ref_line.slope * (x - left_corner[0]) + left_corner[1]
    right_line = lambda x: ref_line.slope * (x - right_corner[0]) + right_corner[1]
    def in_trap(coord):
        # If the sides slope right, then the coord should be below the left line and above the right line; otherwise it is switched
        bt_top_bot = bottom[1] > coord[1] > top[1]
        return bt_top_bot and (left_line(coord[0]) > coord[1] > right_line(coord[0]) or (left_line(coord[0]) < coord[1] < right_line(coord[0])))
    return in_trap

### TRIANGLE SECTION ###
def find_triangles(lines, min_line_length=40, dist_threshold=55, max_triangles=float('inf')):
    """ Given a list of consolidated lines, return any large triangles within them.
    Destructively removes any lines which are a part of the triangles."""
    def closest_line(coord, list_of_lines):
        """ Returns the line from list_of_lines with an endpoint closest to coord.
        If there is no line with a sufficiently close endpoint, return None."""
        closest_endpoint_dist = lambda line: min([Line.coord_dist(coord, c) for c in line.coords])
        closest_line = min(list_of_lines, key=closest_endpoint_dist)
        if closest_endpoint_dist(closest_line) <= dist_threshold:
            return closest_line
        else:
            return None

    def which_side(ref_line, test_line):
        """Returns which side the test line is on relative to the reference line.
        If the reference line is horizontal, then the test line could be 'up' or 'down'.
        If --------------------- vertical, ----------------------------'left' or 'right'."""
        if ref_line.angle_type == 'horz':
            dy1, dy2 = test_line.coords[0][1] - ref_line.center_coord[1], test_line.coords[1][1] - ref_line.center_coord[1]
            farther = max([dy1, dy2], key=abs)
            if farther > 0:
                return 'up'
            else:
                return 'down'
        elif ref_line.angle_type == 'vert':
            dx1, dx2 = test_line.coords[0][0] - ref_line.center_coord[0], test_line.coords[1][0] - ref_line.center_coord[0]
            farther = max([dx1, dx2], key=abs)
            if farther > 0:
                return 'right'
            else:
                return 'left'

    def valid_triangle(ref_line, lin1, lin2):
        """ Returns whether the lines form a valid triangle. """
        if (not lin1) or (not lin2):
            return False
        return True


        check = lambda side1, side2: (which_side(ref_line, lin1) == side1 and which_side(ref_line, lin2) == side1 and lin1.angle_type == 'pos-diag' and lin2.angle_type == 'neg-diag') or (which_side(ref_line, lin1) == side2 and which_side(ref_line, lin2) == side2 and lin1.angle_type == 'neg-diag' and lin2.angle_type == 'pos-diag')
        if ref_line.angle_type == 'horz':
            return check('up', 'down')
        elif ref_line.angle_type == 'vert':
            return check('left', 'right')

    triangles, to_remove = [], []
    long_lines = [line for line in lines if line.length >= min_line_length]
    vs = [line for line in long_lines if line.angle_type == 'vert']
    hs = [line for line in long_lines if line.angle_type == 'horz']
    diags = [line for line in long_lines if line.angle_type in ['pos-diag', 'neg-diag']] 
    if not diags:
        return []
    if vs:
        for vert_line in vs:
            top_coord, bottom_coord = max(vert_line.coords, key=lambda c: c[1]), min(vert_line.coords, key=lambda c: c[1])
            top_closest = closest_line(top_coord, diags)
            bottom_closest = closest_line(bottom_coord, diags)
            if valid_triangle(vert_line, top_closest, bottom_closest):
                direction = which_side(vert_line, top_closest)
                triangles.append(Triangle(vert_line, direction))
                for line in (top_closest, bottom_closest, vert_line):
                    to_remove.append(line)
    if hs:
        for horz_line in hs:
            left_coord, right_coord = max(horz_line.coords, key=lambda c: c[0]), min(vert_line.coords, key=lambda c: c[0])
            left_closest, right_closest = closest_line(left_coord, diags), closest_line(right_coord, diags)
            if valid_triangle(horz_line, left_closest, right_closest):
                direction = which_side(horz_line, left_closest)
                triangles.append(Triangle(horz_line, direction))
                for line in (left_closest, right_closest, horz_line):
                    to_remove.append(line)
    for removable in set(to_remove):
        lines.remove(removable)
    # Return the largest triangles up to the specified maximum number
    return sorted(triangles, key=lambda t: t.size)[0:max_triangles]
            



### CIRCLE SECTION ###

def find_circles(file_name, show_circles=False):
    """Performs the full process of identifying circles in an image.
    file_name is a string pointing to the location of a parseable image
    
    Returns a list of Circles. 
    """
    return find_shape(file_name, 'circle', show_shapes=show_circles)

def circle_hough(image, max_circles=float('inf')):
    """ Uses scikit-image's Hough circle transform to identify circles in an image.

    image: numpy image including circles
    
    Returns a list of filtered Circles
    """
    # Find edges
    edges = feature.canny(image, sigma=0.5, low_threshold=0.2, high_threshold=0.3, use_quantiles=True)
    # Detect radii through Hough transform
    min_rad, max_rad = int(0.05 * np.mean(image.shape)), 0.1 * np.mean(image.shape)

    rad_range = np.arange(min_rad, max_rad, (max_rad - min_rad) // 10)
    accum = transform.hough_circle(edges, rad_range)
    filtered_circles = []
    threshold_percent = 0.85
    # If there is a specified number of circles, lower threshold until we get to that number
    if max_circles < float('inf'):
        while len(filtered_circles) < max_circles:
            accums, cx, cy, radii = transform.hough_circle_peaks(accum, rad_range, min_xdistance=100, min_ydistance=100, threshold=threshold_percent*np.max(accum))
            all_circles = [Circle(cx[i], cy[i], radii[i]) for i in range(len(cx))]
            filtered_circles = filter_circles(all_circles, max_circles)
            threshold_percent = threshold_percent * 0.9
    else:
        accums, cx, cy, radii = transform.hough_circle_peaks(accum, rad_range, min_xdistance=100, min_ydistance=100, threshold=0.85*np.max(accum))
        all_circles = [Circle(cx[i], cy[i], radii[i]) for i in range(len(cx))]
        filtered_circles = filter_circles(all_circles, max_circles)
    for circ in all_circles:
        print(circ)
    return filtered_circles

def filter_circles(list_of_circles, num_circles, d=150):
    """ min_xdistance and min_ydistance of hough_circle_peaks do not work; this substitutes for that functionality

    Returns a list (cxs, cys, radii) where all centers are at least d distance apart.
    Prioritizes the largest circle in each cluster of possible circles.
    """
    def filter_helper(lst):
        if not lst:
            return []
        first_circle, max_circle = lst[0], lst[0]
        for circ in lst[1:]:
            if first_circle.distance_to(circ) <= d:
                if circ.radius > max_circle.radius:
                    lst.remove(max_circle)
                    max_circle = circ
                else:
                    lst.remove(circ)
        lst.remove(max_circle)
        final = [max_circle] + filter_helper(lst)
        return final
    spaced_circles = filter_helper(list_of_circles)
    return sorted(spaced_circles, key=lambda circ: circ.radius)[0:num_circles]
    
simple = './ex_circuits/easy_circuit.jpg'
two_circles = './ex_circuits/two_circles.jpg'
three_circles = './ex_circuits/three_circles.jpg'
integrator = './ex_circuits/integrator.jpg'
better_integrator = './ex_circuits/better_integrator.jpg'
differentiator = './ex_circuits/differentiator.jpg'
printed_simple = './ex_circuits/printed_simple.jpg'
printed_oa = './ex_circuits/printed_oa.jpg'
printed_oa_current = './ex_circuits/printed_oa_current.jpg'

# UNCOMMENT THIS LINE TO TEST (you can replace the file name with the others if you prefer)
image = find_all(better_integrator, printed=False, num_sources=1)
