class Vector():
    
    def __init__(self, x=0, y=0, z=0):
        
        self.x = x
        self.y = y
        self.z = z
    
    def __eq__(self, other):
        
        if not isinstance(other, Vector):
            return False
        
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def copy(self):
        
        return Vector(self.x, self.y, self.z)
    
    def __repr__(self):
        
        return '%.3f, %.3f, %.3f' % (self.x, self.y, self.z)
    
    def dist(self, v):
        
        xd = self.x - v.x
        yd = self.y - v.y
        zd = self.z - v.z
        
        return math.sqrt(xd*xd + yd*yd + zd*zd)
    
    @property
    def length(self):
        return math.sqrt(self.x*self.x+self.y*self.y+self.z*self.z)

    @property
    def unit(self):
        l=self.length
        if l==0: return self
        return self.divide(l)
    
    def add(self,vector):
        self.x+=int(vector.x)
        self.y+=int(vector.y)
        self.z+=int(vector.z)
        return self
    
    def divide(self,factor):
        self.x/=factor
        self.y/=factor
        self.z/=factor
        return self

    def __add__(self, other):
        vector = Vector(self.x, self.y, self.z)
        return vector.add(other)
    
    def scale(self,factor):
        self.x*=factor
        self.y*=factor
        self.z*=factor
        return self
    
    def __mul__(self, factor):
        vector = Vector(self.x, self.y, self.z)
        return vector.scale(factor)

    # for python 3
    def __truediv__(self, factor):
        vector = Vector(self.x, self.y, self.z)
        return vector.divide(factor)

    # for python 2
    def __div__(self, factor):
        vector = Vector(self.x, self.y, self.z)
        return vector.divide(factor)

class Bound():
    
    def __init__(self, x_len=0, y_len=0, z_len=1):
        
        self.x_len = x_len
        self.y_len = y_len
        self.z_len = z_len
    
    def is_inside(self, vector):
        
        if 0 <= vector.x < self.x_len and 0 <= vector.y < self.y_len and 0<= vector.z < self.z_len:
            
            return True
        
        else:
            
            return False