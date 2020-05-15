""" A pool of operations that keep grad of the gradient.  """

import numpy as np


class _Operation:
    """ Operation abstract class. 

    An operations takes two operands and computes the result in the 'forward'
    method and the gradient in the 'backward' method.
    
    Attributes
    ----------
    a : object
        First operand of the operation.
    b : object
        Second operand of the operation.
    a_grad : object
        Gradient of the first operand.
    b_grad : object
        Gradient of the second operand
    
    """
    def __init__(self):
        # operands
        self.a = None
        self.b = None
        
        # gradient of the operands
        self.a_grad = None
        self.b_grad = None
    
    def forward(self):
        pass

    def backward(self, incoming_grad=1):
        pass


class Add(_Operation):
    """Adds.
    
    Implements the operation a + b and computes its gradient.
    
    """
    def __init__(self):
        super(Add, self).__init__()
    
    def forward(self, a, b):
        self.a = a
        self.b = b
        
        return np.add(a,b)
    
    def backward(self, incoming_grad=1):
        self.a_grad = incoming_grad * 1
        self.b_grad = incoming_grad * 1
        
        return self.a_grad, self.b_grad
    

class Subtract(_Operation):
    """Subtracts.
    
    Implements the operation a - b and computes its gradient.
    
    """
    def __init__(self):
        super(Subtract, self).__init__()
    
    def forward(self, a, b):
        self.a = a
        self.b = b
        
        return np.subtract(a,b)
    
    def backward(self, incoming_grad=1):
        self.a_grad = incoming_grad * 1
        self.b_grad = incoming_grad * (-1)
        
        return self.a_grad, self.b_grad

    
class Multiply(_Operation):
    """Multiplies.
    
    Implements the operation a * b and computes its gradient.
    
    """
    def __init__(self):
        super(Multiply, self).__init__()
    
    def forward(self, a, b):
        self.a = a
        self.b = b
        
        return np.multiply(a,b)
    
    def backward(self, incoming_grad=1):
        self.a_grad = incoming_grad * self.b
        self.b_grad = incoming_grad * self.a
        
        return self.a_grad, self.b_grad


class Divide(_Operation):
    """Divides.
    
    Implements the operation a / b and computes its gradient.
    
    """
    def __init__(self):
        super(Divide, self).__init__()
        
    def forward(self, a, b):
        self.a = a
        self.b = b
        
        return np.divide(a, b)
    
    def backward(self, incoming_grad=1):
        self.a_grad = incoming_grad * (1 / self.b)
        self.b_grad = incoming_grad * (-(self.a / (self.b)**2))
        
        return self.a_grad, self.b_grad

    
class Pow(_Operation):
    """Pow.
    
    Implements the operation a^b and computes its gradient.
    
    """
    def __init__(self):
        super(Pow, self).__init__()
    
    def forward(self, a, b):
        self.a = a
        self.b = b
        
        return self.a ** self.b
    
    def backward(self, incoming_grad=1):
        self.a_grad = incoming_grad * self.b * (self.a ** (self.b - 1))
        self.b_grad = incoming_grad * np.log(self.a) * (self.a ** self.b)
        
        return self.a_grad, self.b_grad


class Sin(_Operation):
    """Sin.

    Implements the operation sin(a) and computes its gradient.

    """
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, a):
        self.a = a

        return np.sin(self.a)

    def backward(self, incoming_grad=1):
        self.a_grad = np.cos(self.a) * incoming_grad
        
        return self.a_grad


class Cos(_Operation):
    """Cos.

    Implements the operation cos(a) and computes its gradient.

    """
    def __init__(self):
        super(Cos, self).__init__()

    def forward(self, a):
        self.a = a

        return np.cos(self.a)

    def backward(self, incoming_grad=1):
        self.a_grad = - np.sin(self.a) * incoming_grad
        
        return self.a_grad


class Tan(_Operation):
    """Tan.

    Implements the operation tan(a) and computes its gradient.

    """
    def __init__(self):
        super(Tan, self).__init__()
    
    def forward(self, a):
        self.a = a

        return np.tan(self.a)

    def backward(self, incoming_grad=1):
        self.a_grad = incoming_grad / (np.cos(self.a) ** 2)

        return self.a_grad