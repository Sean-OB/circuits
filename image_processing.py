import matplotlib.pyplot as plt
import imageio as img
import numpy as np
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.filters import median

def find_circles(file_name, show_circles=False):
    """Performs the full process of identifying circles in an image.
    file_name is a string pointing to the location of a parseable image
    
    Returns a list of Circles. 
    """
    image = bw_image(file_name)
    color_image = plt.imread(file_name)
    color_image.setflags(write=1)
    circles = circle_hough(image)
    if show_circles:
        for circ in circles:
            cy, cx = circle_perimeter(circ.y, circ.x, circ.radius)
            color_image[cy, cx] = (220, 20, 20)
        show(color_image)
    return circles


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
    return contrastify(median(img.imread(file_name, as_gray=True).astype('int')))

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

def circle_hough(image, max_circles=10):
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


simple = './ex_circuits/easy_circuit.jpg'
two_circles = './ex_circuits/two_circles.jpg'
three_circles = './ex_circuits/three_circles.jpg'

# UNCOMMENT THIS LINE TO TEST (you can replace the file name with the others if you prefer)
#find_circles(three_circles, show_circles=True)
