import matplotlib.pyplot as plt
import imageio as img
import numpy as np
from skimage import draw, feature, filters, morphology, transform, util

def find_all(file_name, num_sources=10, num_oas=5):
    """Main function of this file. Identifies all structures in an image and returns them
    data structure: TODO """
    image = bw_image(file_name)
    circles = circle_hough(image, num_sources)
    lines = line_hough(image)
    remove_close_to_circles(lines, circles)
    filtered_lines = consolidate_lines(lines, bw_image(file_name), circles)
    triangles = find_triangles(filtered_lines)
    
    drawn_image = draw_shapes([filtered_lines, circles, triangles], file_name)
    show(drawn_image)
    return triangles


### MISCELLANEOUS HELPER FUNCTIONS ###

def bw_image(file_name):
    """
    Returns a black and white, contrastified, eroded, smoothed image.

    file_name: string file name of an image file to be read."""
    processed = morphology.erosion(filters.median(img.imread(file_name, as_gray=True).astype('uint8')))
    x_size, y_size = processed.shape
    # Split up to deal with shadows and changes in light
    quad1, quad2, quad3, quad4 = processed[0:x_size//2, 0:y_size//2], processed[x_size//2:x_size, 0:y_size//2], processed[0:x_size//2, y_size//2:y_size], processed[x_size//2:x_size, y_size//2:y_size]
    quad1, quad2, quad3, quad4 = contrastify(quad1), contrastify(quad2), contrastify(quad3), contrastify(quad4)
    return np.vstack((np.hstack((quad1, quad3)), np.hstack((quad2, quad4))))

def contrastify(arr, midpoint=None):
    """
    Takes a numpy array and returns a new one separated at a midpoint, mapping smaller values to zero and larger ones to a maximum value.

    image: numpy array of numbers
    midpoint: lower values are mapped to zero, equal or higher values are mapped to max_val
    """
    if not midpoint:
        midpoint = (np.min(arr) + np.max(arr)) * 0.55
    arr = (arr > midpoint) * 255
    return arr

def draw_shapes(list_of_shapes, file_name):
    """ Draws all given shapes onto an image.
    list_of_shapes is a list of shape lists (e.g. a [[all circles], [all lines]]), whose members have a draw function"""
    color_image = plt.imread(file_name)
    color_image.setflags(write=1)
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

### POLISHING FUNCTIONS ###

def remove_close_to_circles(lines, circles, min_rat=0.6, max_rat=1.4):
    """ Destructively removes any line that with both endpoints close to the edge of a circle. """
    to_remove = []
    for line in lines:
        if any([circ.coord_close_to_circle(line.coords[0], min_rat, max_rat) and circ.coord_close_to_circle(line.coords[1], min_rat, max_rat) for circ in circles]):
            to_remove.append(line)
    [lines.remove(line) for line in to_remove]

### LINE SECTION ###

def find_lines(file_name, show_lines=False):
    """Performs the full process of identifying lines in an image."""
    return find_shape(file_name, 'line', show_shapes=show_lines)

def line_hough(image):
        
    # Find edges
    edges = feature.canny(image, sigma=0.5, low_threshold=0.1, high_threshold=0.2, use_quantiles=True)
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
        if len(list_of_lines) == 1:
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
            look = rect(ref_line)
        else:
            look = trap(ref_line)
        closest = min([line for line in list_of_lines if line.angle_type == ref_line.angle_type and (look(line.coords[0]) or look(line.coords[1]))], key=lambda lin: ref_line.min_endpoint_dist(lin), default=None)
        return closest


    def rect(ref_line):
        """ Returns a rectangular look function based on the reference line."""
        if ref_line.angle_type == 'horz':
            # Rectangle that is width wide from the center of the line in the direction of the first endpoint
            right_x, left_x= max(ref_line.x0, ref_line.x1), min(ref_line.x0, ref_line.x1)
            # look is a function that takes coordinates and returns whether they are in the rectangle
            bl_corner = (left_x - dist, ref_line.center[1] - width)
            tr_corner = (right_x + dist, ref_line.center[1] + width)
        elif ref_line.angle_type == 'vert':
            top_y, bottom_y  = max(ref_line.y0, ref_line.y1), min(ref_line.y0, ref_line.y1)
            bl_corner = (ref_line.center[0] - width, bottom_y - dist)
            tr_corner = (ref_line.center[0] + width, top_y + dist)
        def in_rect(coord):
            """ Determines if coord is within a rectangle spanned by 2 given corners. All coordinates are given in tuples. """
            return bl_corner[0] < coord[0] < tr_corner[0] and bl_corner[1] < coord[1] < tr_corner[1]
        return in_rect

    def trap(ref_line):
        """ Returns a trapezoidal look function based on the reference line.
        NOTE: higher y values are lower, due to the nature of numpy arrays.
        As such, bottom refers to the LOWER of the two points and so on"""

        # Scale width down for diagonal lines
        diag_width = width // 8
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

    # Actual body of the function
    dist, width, min_proportion_filled = int(0.02 * np.mean(image.shape)), int(0.03 * np.mean(image.shape)), 0.2
    filtered_lines = consolidate_helper(lines)
    return filtered_lines

### TRIANGLE SECTION ###
def find_triangles(lines, min_line_length=40, dist_threshold=55):
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
            dy1, dy2 = test_line.coords[0][1] - ref_line.center[1], test_line.coords[1][1] - ref_line.center[1]
            farther = max([dy1, dy2], key=abs)
            if farther > 0:
                return 'up'
            else:
                return 'down'
        elif ref_line.angle_type == 'vert':
            dx1, dx2 = test_line.coords[0][0] - ref_line.center[0], test_line.coords[1][0] - ref_line.center[0]
            farther = max([dx1, dx2], key=abs)
            if farther > 0:
                return 'right'
            else:
                return 'left'

    def valid_triangle(ref_line, lin1, lin2):
        """ Returns whether the lines form a valid triangle. """
        if (not lin1) or (not lin2):
            return False

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
    for vert_line in vs:
        top_coord, bottom_coord = max(vert_line.coords, key=lambda c: c[1]), min(vert_line.coords, key=lambda c: c[1])
        top_closest = closest_line(top_coord, diags)
        bottom_closest = closest_line(bottom_coord, diags)
        if valid_triangle(vert_line, top_closest, bottom_closest):
            direction = which_side(vert_line, top_closest)
            triangles.append(Triangle(vert_line, direction))
            for line in (top_closest, bottom_closest, vert_line):
                to_remove.append(line)
#                lines.remove(line)
    for horz_line in hs:
        left_coord, right_coord = max(horz_line.coords, key=lambda c: c[0]), min(vert_line.coords, key=lambda c: c[0])
        left_closest, right_closest = closest_line(left_coord, diags), closest_line(right_coord, diags)
        if valid_triangle(horz_line, left_closest, right_closest):
            direction = which_side(horz_line, left_closest)
            triangles.append(Triangle(horz_line, direction))
            for line in (left_closest, right_closest, horz_line):
                to_remove.append(line)
#                lines.remove(line)
    for removable in set(to_remove):
        lines.remove(removable)
    return triangles
            



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

    rad_range = np.arange(min_rad, max_rad, (max_rad - min_rad) // 20)
    accum = transform.hough_circle(edges, rad_range)
#    accums, cx, cy, radii = transform.hough_circle_peaks(accum, rad_range, min_xdistance=100, min_ydistance=100, threshold=0.85*np.max(accum), num_peaks=max_circles)
    accums, cx, cy, radii = transform.hough_circle_peaks(accum, rad_range, min_xdistance=100, min_ydistance=100, threshold=0.85*np.max(accum))
    all_circles = [Circle(cx[i], cy[i], radii[i]) for i in range(len(cx))]
    filtered_circles = filter_circles(all_circles, max_circles)
    print(len(filtered_circles))
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
    return filter_helper(list_of_circles)[0:num_circles]
    
class Circle:
    """A circle has coordinates and a radius."""

    def __init__(self, x, y, radius):
        self.x = int(x)
        self.y = int(y)
        self.radius = int(radius)

    @property
    def center_coord(self):
        """ Returns a tuple of the center coordinates of the circle. """
        return (self.x, self.y)

    def __str__(self):
        return 'x: {0}, y: {1}, r: {2}'.format(self.x, self.y, self.radius)

    def distance_to(self, circ2):
        return np.sqrt((self.x - circ2.x) ** 2 + (self.y - circ2.y) ** 2)

    def coord_close_to_circle(self, coord, min_rad_rat, max_rad_rat):
        """ Returns whether a coordinate is between min_rad_rat * radius and max_rad_rad * radius of the center """
        assert max_rad_rat >= min_rad_rat, 'Maximum ratio must be larger or equal to the minimum ratio!'
        return self.radius * min_rad_rat < Line.coord_dist(coord, self.center_coord) < self.radius * max_rad_rat

    def draw(self, image, color=(220, 20, 20)):
        """ Destructively draws the circle onto a given image. """
        cy, cx = draw.circle_perimeter(self.y, self.x, self.radius)
        image[cy, cx] = color

class Coord:
    """ A 2-dimensional coordinate. """
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Line:
    """ A line segment expressed by the Cartesian coordinates of its endpoints.
    Note that the visualized version of a line will be flipped vertically, so positive diagonal will look like negative diagonal."""

    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def __str__(self):
        return '({0}, {1}), ({2}, {3})'.format(self.x0, self.y0, self.x1, self.y1)

    @property
    def angle(self):
        if self.x0 - self.x1 == 0:
            return 90
        else:
            raw_angle = np.arctan((self.y0 - self.y1) / (self.x0 - self.x1)) * 180 / np.pi
            if -90 <= raw_angle < -67.5 or raw_angle >= 67.5:
                return 90
            elif -67.5 <= raw_angle < -22.5:
                return -45
            elif -22.5 <= raw_angle < 22.5:
                return 0
            elif 22.5 <= raw_angle < 67.5:
                return 45

    @property
    def slope(self):
        if self.x0 - self.x1 == 0:
            return float('inf')
        else:
            return (self.y0 - self.y1) / (self.x0 - self.x1)

    @property
    def angle_type(self):
        if self.angle == 90:
            return 'vert'
        elif -67.5 <= self.angle < -22.5:
            return 'neg-diag'
        elif -22.5 <= self.angle < 22.5:
            return 'horz'
        elif 22.5 <= self.angle < 67.5:
            return 'pos-diag'

    @property
    def coords(self):
        """ Returns the coordinates of endpoints in the format ((x0, y0), (x1, y1)) """
        return ((self.x0, self.y0), (self.x1, self.y1))

    @property
    def center(self):
        return ((self.x0 + self.x1) // 2, (self.y0 + self.y1) // 2)

    @property
    def length(self):
        """ Returns the Euclidean length of the line """
        dx, dy = self.x1 - self.x0, self.y1 - self.y0
        return np.sqrt(dx ** 2 + dy ** 2)

    def from_polar(d0, a0, d1, a1, starting=(0, 0)):
        """ Angles are in radians """
        x0 = starting[0] + d0 * np.cos(a0)
        y0 = starting[1] + d0 * np.sin(a0)
        x1 = starting[0] + d1 * np.cos(a1)
        y1 = starting[1] + d1 * np.sin(a1)
        return Line(x0, y0, x1, y1)

    def from_coords(c0, c1):
        """ Coordinates are an unpackable pair, preferably a tuple """
        x0, y0 = c0
        x1, y1 = c1
        return Line(x0, y0, x1, y1)

    def min_endpoint_dist(self, line):
        """ Returns the minimum distance between the endpoints of two lines """
        dists = []
        for i in range(2):
            for j in range(2):
                dists.append(Line.coord_dist(self.coords[i], line.coords[j]))
        return min(dists)

    def join_lines(self, line):
        """ Returns the combination of two lines together at their closest endpoint """
        # Farthest endpoints are formatted as a tuple of coordinate tuples
        farthest_endpoints = max([self.coords, line.coords], key=lambda cs: Line.coord_dist(cs[0], cs[1]))
        farthest_distance = Line.coord_dist(farthest_endpoints[0], farthest_endpoints[1])
        assert self.angle_type == line.angle_type, 'Line angles must be of the same type!'
        for i in range(2):
            for j in range(2):
                if Line.coord_dist(self.coords[i], line.coords[j]) > farthest_distance:
                    farthest_endpoints = (self.coords[i], line.coords[j])
                    farthest_distance = Line.coord_dist(self.coords[i], line.coords[j])
        
        if self.angle_type == 'horz':
            avg_y = (farthest_endpoints[0][1] + farthest_endpoints[1][1]) // 2
            return Line(farthest_endpoints[0][0], avg_y, farthest_endpoints[1][0], avg_y)
        elif self.angle_type == 'vert':
            avg_x = (farthest_endpoints[0][0] + farthest_endpoints[1][0]) // 2
            return Line(avg_x, farthest_endpoints[0][1], avg_x, farthest_endpoints[1][1])
        else:
            return Line.from_coords(farthest_endpoints[0], farthest_endpoints[1])
        

    def coord_dist(c0, c1):
        """ c0 and c1 are coordinate tuples of the form (x, y) """
        return np.sqrt((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2)

    def draw(self, image, color=(220, 20, 20)):
        """ Destructively draws line onto an image. """
        image[draw.line(self.y0, self.x0, self.y1, self.x1)] = color



class Triangle:
    """ Represents an equilateral triangle in coordinate space, with a base perp/parallel to the two elementary axes of 2-D space. """
    def __init__(self, base_line, direction):
        # Note: direction marks where the vertex opposite the base line points (left, right, up or down)
        self.base_line = base_line
        self.direction = direction
        if base_line.angle_type == 'horz':
            assert direction in ['up', 'down'], 'Direction must be perpendicular to the base line'
        elif base_line.angle_type == 'vert':
            assert direction in ['left', 'right'], 'Direction must be perpendicular to the base line'
        else:
            raise AssertionError('Invalid direction')
        if self.direction == 'up':
            self.vertex = (self.base_line.center[0], int(self.base_line.center[1] - self.base_line.length * np.sqrt(3) / 2))
        elif self.direction == 'down':
            self.vertex = (self.base_line.center[0], int(self.base_line.center[1] + self.base_line.length * np.sqrt(3) / 2))
        elif self.direction == 'left':
            self.vertex = (self.base_line.center[0] - int(self.base_line.length * np.sqrt(3) / 2), self.base_line.center[1])
        elif self.direction == 'right':
            self.vertex = (self.base_line.center[0] + int(self.base_line.length * np.sqrt(3) / 2), self.base_line.center[1])

    def draw(self, image, color=(220, 20, 20)):
        """ Destructively draws the triangle onto the image. """
        self.base_line.draw(image)
        Line.from_coords(self.vertex, self.base_line.coords[0]).draw(image)
        Line.from_coords(self.vertex, self.base_line.coords[1]).draw(image)
        


simple = './ex_circuits/easy_circuit.jpg'
two_circles = './ex_circuits/two_circles.jpg'
three_circles = './ex_circuits/three_circles.jpg'
integrator = './ex_circuits/integrator.jpg'
better_integrator = './ex_circuits/better_integrator.jpg'
differentiator = './ex_circuits/differentiator.jpg'

# UNCOMMENT THIS LINE TO TEST (you can replace the file name with the others if you prefer)
# triangles = find_all(better_integtrator)
