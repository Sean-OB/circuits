from sympy import *

class Element:
    """ Master class that includes the attributes of each circuit element. All circuit elements inherit from this class.
    
    coord: coordinate tuple of the element, e.g. (245, 363)
    name: unique name of the element (e.g. r3 for 3rd resistor, w5 for 5th wire)
    v_i: voltage at the entrance to the element
    v_o: voltage at the exit of the element
    curr: current through the element (same at beginning and end according to KCL)
    res: resistance of the element
    """
    def __init__(self, coord, name, v_i=(), v_o=(), curr=(), res=()):
        self.coord = coord
        self.name = name
        self.v_i = v_i if v_i else self.symbolic_name('vi')
        self.v_o = v_o if v_o else self.symbolic_name('vo')
        self.curr = curr if curr else self.symbolic_name('i')
        self.res = res if res else self.symbolic_name('r')
        self.inputs, self.outputs = [], []

    def link_to_output(self, el2):
        """ Links this element to one of its outputs
        Used during the initial traversal of the system."""
        self.add_output(el2)
        el2.add_input(self)

    def link_to_input(self, el2):
        """ Links this element to one of its inputs
        Used during the initial traversal of the system."""
        self.add_input(el2)
        el2.add_input(self)

    def add_output(self, el2):
        """ Adds a single output to this element """
        self.outputs.append(el2)

    def add_input(self, el2):
        """ Adds a single input to this element """
        self.inputs.append(el2)

    def symbolic_name(self, quantity):
        """ Returns a unique symbolic name for a given quantity of an element """
        return Symbol(quantity + '_{' + self.name[0] + '_' + self.name[1] + '}')

    def __str__(self):
        output_string = ''
        output_string += 'Name: ' + self.name + '\n'
        output_string += 'Input voltage: ' + str(self.v_i) + '\n'
        output_string += 'Output voltage: ' + str(self.v_o) + '\n'
        output_string += 'Current: ' + str(self.curr) + '\n'
        output_string += 'Resistance: ' + str(self.res)
        return output_string


