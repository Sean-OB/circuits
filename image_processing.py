import matplotlib.pyplot as plt
import imageio as img
import numpy as np
import cv2 as cv
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks, probabilistic_hough_line, hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter, line
from skimage.util import img_as_ubyte
from skimage.filters import median

def find_all(file_name, show_shapes=False):
    """Main function of this file. Identifies all structures in an image and returns them
    data structure: TODO """
    find_shape(file_name, 'circle', show_shapes=show_shapes)



def contrastify(arr, midpoint=None):
    """
    Takes a numpy array and returns a new one separated at a midpoint, mapping smaller values to zero and larger ones to a maximum value.

    image: numpy array of numbers
    midpoint: lower values are mapped to zero, equal or higher values are mapped to max_val
    """
    if not midpoint:
        midpoint = (np.min(arr) + np.max(arr)) / 2
    return (arr > midpoint) * 255

def bw_image(file_name):
    """
    Returns a black and white, contrastified, smoothed image.

    file_name: string file name of an image file to be read."""
    image = img.imread(file_name, as_gray=True)
    return contrastify(img.imread(file_name, as_gray=True)).astype('uint8')

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
    if show_shapes:
        disp(shapes, color_image)
    return shapes
        

### LINE SECTION ###

def find_lines(file_name, show_lines=False):
    """Performs the full process of identifying lines ina n image.
    file_name is a string pointing to the location of a parseable image

    Returns a list of Lines.
    """
    return find_shape(file_name, 'line', show_shapes=show_lines)

def line_hough(image):
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


def disp_lines(lines, color_image):
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
    image_bytes = median(img_as_ubyte(image))
    edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)

    # Detect radii through Hough transform
    rad_range = np.arange(50, 500, 10)
    accum = hough_circle(edges, rad_range)
    accums, cx, cy, radii = hough_circle_peaks(accum, rad_range, min_xdistance=100, min_ydistance=100, threshold=0.85*np.max(accum), num_peaks=max_circles)
    all_circles = [Circle(cx[i], cy[i], radii[i]) for i in range(len(cx))]
    return filter_circles(all_circles)

def disp_circles(circles, color_image):
    for circ in circles:
        cy, cx = circle_perimeter(circ.y, circ.x, circ.radius)
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

    def from_polar(d0, a0, d1, a1, starting=(0, 0)):
        """ Angles are in radians """
        x0 = starting[0] + d0 * np.cos(a0)
        y0 = starting[1] + d0 * np.cos(a0)
        x1 = starting[0] + d1 * np.cos(a1)
        y1 = starting[1] + d1 * np.cos(a1)
        return Line(x0, y0, x1, y1)


simple = './ex_circuits/easy_circuit.jpg'
two_circles = './ex_circuits/two_circles.jpg'
three_circles = './ex_circuits/three_circles.jpg'

# UNCOMMENT THIS LINE TO TEST (you can replace the file name with the others if you prefer)
# find_all(simple, show_shapes=True)
find_lines(simple, show_lines=True)
