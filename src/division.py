import numpy as np

def div(x,y):
    if isinstance(x,int) and isinstance(y,int):
        return x//y
    else:
        return x/y
