import matplotlib.pyplot as plt
import imageio as img
import numpy as np
import cv2 as cv
import skimage
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks, probabilistic_hough_line, hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.util import img_as_ubyte
from skimage.filters import median

def find_all(file_name, show_shapes=False):
    """Main function of this file. Identifies all structures in an image and returns them
    data structure: TODO """
    find_shape(file_name, 'circle', show_shapes=show_shapes)

### MISCELLANEOUS HELPER FUNCTIONS ###

def bw_image(file_name):
    """
    Returns a black and white, contrastified, eroded, smoothed image.

    file_name: string file name of an image file to be read."""
    processed = skimage.morphology.erosion(median(img.imread(file_name, as_gray=True).astype('uint8'), selem=np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])))
    contrastified = contrastify(processed)
    return contrastified

def contrastify(arr, midpoint=None):
    """
    Takes a numpy array and returns a new one separated at a midpoint, mapping smaller values to zero and larger ones to a maximum value.

    image: numpy array of numbers
    midpoint: lower values are mapped to zero, equal or higher values are mapped to max_val
    """
    if not midpoint:
        midpoint = (np.min(arr) + np.max(arr)) * 0.5
    arr = (arr > midpoint) * 255
    return arr

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

### SHAPE FINDING FUNCTIONS ###

def find_shape(file_name, shape, show_shapes=False):
    """ Finds all shapeures in an image. Types of structurs:
    circles: 'circle'
    lines: 'line'
    """
    image = bw_image(file_name)
    color_image = plt.imread(file_name)
    color_image.setflags(write=1)
    if shape == 'circle':
        shapes = circle_hough(image)
        disp = disp_circles
    elif shape == 'line':
        shapes = line_hough(image)
        disp = disp_lines
    elif shape == 'corner':
        shapes = corner_detection(image)
        disp = disp_corners
    if show_shapes:
        disp(shapes, color_image)
    return shapes

### LINE SECTION ###

def find_lines(file_name, show_lines=False):
    """Performs the full process of identifying lines in an image."""
    return find_shape(file_name, 'line', show_shapes=show_lines)

def line_hough(image):
        
    # Find edges
    """
    image_bytes = median(img_as_ubyte(image))
    edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)
    """
    edges = canny(image, sigma=0.5, low_threshold=0.2, high_threshold=0.3, use_quantiles=True)
    line_coords = probabilistic_hough_line(edges, threshold=5, line_length=50, line_gap=15, theta=np.pi * np.arange(-1/2, 1/2, 0.01))
    lines = [Line(c0[0], c0[1], c1[0], c1[1]) for c0, c1 in line_coords]
    return lines

def disp_lines(lines, color_image):
    for line in lines:
        color_image[skimage.draw.line(line.y0, line.x0, line.y1, line.x1)] = (220, 20, 20)
    print(len(lines))
    show(color_image)

def consolidate_lines(lines, circles):
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

### CORNER SECTION ###

def find_corners(file_name, show_corners=False):
    """Performs the full process of identifying lines in an image.
    file_name is a string pointing to the location of a parseable image

    Returns a list of Lines.
    """
    return find_shape(file_name, 'corner', show_shapes=show_corners)



def corner_detection(image):
    """ Uses scikit-image's probabilistic Hough line transform to identify lines in an image.

    image: numpy image

    Returns a list of Lines
    """
    edges = cv.Canny(image, 50, 150, apertureSize=3).astype('uint8')
    accum = cv.cornerHarris(edges, 20, 31, 0.02)
    print(accum)
    print(np.count_nonzero(accum > 0.15 * np.max(accum)))
    return accum
    

def filter_lines(hspace, angles, distances):
    """ Threshold parameter on scikit probabilistic_hough_line does not work.
    This is a workaround to fit the project (and extra practice with Hough transforms)
    hspace: 
    """
    def length(line):
        return np.sqrt((line[0] - line[2]) ** 2 + (line[1] - line[3]) ** 2)


def disp_corners(lines, color_image):
    """ Displays the detected line segments """
    color_image[lines > 0.15 * np.max(lines)] = (220, 20, 20)
    show(color_image)
    """
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(color_image, (x1, y1), (x2, y2), color=(220, 20, 20), thickness=5)
    show(color_image)
    """

    

### CIRCLE SECTION ###

def find_circles(file_name, show_circles=False):
    """Performs the full process of identifying circles in an image.
    file_name is a string pointing to the location of a parseable image
    
    Returns a list of Circles. 
    """
    return find_shape(file_name, 'circle', show_shapes=show_circles)

def circle_hough(image, max_circles=15):
    """ Uses scikit-image's Hough circle transform to identify circles in an image.

    image: numpy image including circles
    
    Returns a list of filtered Circles
    """
    # Find edges
    edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)

    # Detect radii through Hough transform
    rad_range = np.arange(50, 500, 10)
    accum = hough_circle(edges, rad_range)
    accums, cx, cy, radii = hough_circle_peaks(accum, rad_range, min_xdistance=100, min_ydistance=100, threshold=0.85*np.max(accum), num_peaks=max_circles)
    all_circles = [Circle(cx[i], cy[i], radii[i]) for i in range(len(cx))]
    return filter_circles(all_circles)

def disp_circles(circles, color_image):
    for circ in circles:
        cy, cx = skimage.draw.circle_perimeter(circ.y, circ.x, circ.radius)
        color_image[cy, cx] = (220, 20, 20)
    show(color_image)

def filter_circles(list_of_circles, d=150):
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
    return filter_helper(list_of_circles)
    
class Circle:
    """A circle has coordinates and a radius."""

    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def __str__(self):
        return 'x: {0}, y: {1}, r: {2}'.format(self.x, self.y, self.radius)

    def distance_to(self, circ2):
        return np.sqrt((self.x - circ2.x) ** 2 + (self.y - circ2.y) ** 2)

class Line:
    """ A line segment expressed by the Cartesian coordinates of its endpoints"""

    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        if self.x0 - self.x1 == 0:
            self.angle = 90
            self.type = 'vert'
        else:
            angle = np.arctan((self.y0 - self.y1) / (self.x0 - self.x1)) * 180 / np.pi
            if -90 <= angle < -75 or angle >= 52.5:
                self.angle = 90
                self.type = 'vert'
            elif -75 <= angle < -52.5:
                self.angle = -60
                self.type = 'neg-diag'
            elif -52.5 <= angle < -37.5:
                self.angle = -45
                self.type = 'neg-diag'
            elif -37.5 <= angle < -15:
                self.angle = -30
                self.type = 'neg-diag'
            elif -15 <= angle < 15:
                self.angle = 0
                self.type = 'horz'
            elif 15 <= angle < 37.5:
                self.angle = 30
                self.type = 'pos-diag'
            elif 37.5 <= angle < 52.5:
                self.angle = 45
                self.type = 'pos-diag'
            elif 52.5 <= angle < 75:
                self.angle = 60
                self.type = 'pos-diag'

    def from_polar(d0, a0, d1, a1, starting=(0, 0)):
        """ Angles are in radians """
        x0 = starting[0] + d0 * np.cos(a0)
        y0 = starting[1] + d0 * np.sin(a0)
        x1 = starting[0] + d1 * np.cos(a1)
        y1 = starting[1] + d1 * np.sin(a1)
        return Line(x0, y0, x1, y1)

    def length(self):
        """ Returns the Euclidean length of the line """
        dx, dy = self.x1 - self.x0, self.y1 - self.y0
        return np.sqrt(dx ** 2 + dy ** 2)

    def coords(self):
        """ Returns the coordinates of endpoints in the format ((x0, y0), (x1, y1)) """
        return ((self.x0, self.y0), self.x1, self.y1)

    def endpoint_min_dist(self, line):
        """ Returns the minimum distance between the endpoints of two lines """
        dists = []
        for i in range(2):
            for j in range(2):
                dists += coord_dist(self.coords[i], line.coords[j])
        return min(dists)

    def coord_dist(c0, c1):
        """ c0 and c1 are coordinate tuples of the form (x, y) """
        return np.sqrt((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2)

    def line_min_dist(self, line):
        """ Returns the minimum Euclidean distance from another line to this one.
        Note that the minimum distance is the orthogonal projection of the difference vector onto this line."""
        


simple = './ex_circuits/easy_circuit.jpg'
two_circles = './ex_circuits/two_circles.jpg'
three_circles = './ex_circuits/three_circles.jpg'

# UNCOMMENT THIS LINE TO TEST (you can replace the file name with the others if you prefer)
# find_all(simple, show_shapes=True)
find_lines(three_circles, show_lines=True)
#image = bw_image(simple)
