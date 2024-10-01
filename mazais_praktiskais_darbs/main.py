import math
import random
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    
    
def distance(p1: Point, p2: Point) -> float:
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def main():
    x_max = 100
    y_max = 100
    station = Point(x_max/2, y_max/2)
    points = [Point(random.uniform(0, x_max), random.uniform(0, y_max)) for _ in range(10)]
    points.insert(0, station)  
    [print(p) for p in points]
    
    # Distance matrix
    distance_matrix = [[distance(p1, p2) for p2 in points] for p1 in points]
    print(distance_matrix)
    
    
if __name__ == "__main__":
    main()