from math import floor

class MY_PDF():
    def __init__(self, a=0, b=0):        
        self.buckets = [0]*100
        self.total = 0  
        self.a = a
        self.b = b

    def feed(self, values):
        for value in values:    
            index = int(floor((value / float(self.b - self.a)) * float(100)))
            val = self.buckets[index]
            self.buckets[index] = val + 1
        self.total = len(values)
        self.buckets = map(lambda x: x / float(self.total), self.buckets)