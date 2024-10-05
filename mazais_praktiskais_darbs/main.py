import math
import random
from dataclasses import dataclass
from typing import List, Callable
from enum import Enum
import time as t


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Cost:
    hard: float
    soft: float


class LocationType(Enum):
    customer = "customer"
    station = "station"


@dataclass
class Location(Point):
    name: LocationType


@dataclass
class Route:
    route: List[Location]


def distance(p1: Point, p2: Point) -> float:
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


@dataclass
class Domain:
    station: Location
    customers: List[Location]


@dataclass
class Solution:
    routes: List[Route]


class SimulatedAnnealing:

    def main(
        domain: Domain,
        cost_function: Callable,  # TODO: add strong typing to callable function
        neighbour_function: Callable,  # TODO: add strong typing to callable function
        temperature: List[float],  # TODO: check if type is correct
        iteration_steps: List[int],  # TODO: check if type is correct
    ) -> Solution:
        initial_solution = SimulatedAnnealing.initialize_solution(domain, cost_function)
        # TODO: implement the rest of the algorithm. Can be found in L3 slide 7
        return initial_solution

    @staticmethod
    def initialize_solution(solution_set: Domain, cost_function: Callable) -> Solution:
        # TODO: implement proper random initialization
        res = []
        res.append(solution_set.station)
        res.extend(solution_set.customers)
        res.append(solution_set.station)
        return Solution([res])

    @staticmethod
    def cost_function(solution: Solution) -> Cost:  # TODO: implement cost function
        return Cost(0, 0)

    @staticmethod
    def neighbour_function(
        solution: Solution,
    ) -> List[Solution]:  # TODO: implement neighbor function
        ...


class prints:
    @staticmethod
    def solution(s: Solution) -> None:
        print("Cost:", SimulatedAnnealing.cost_function(s))
        for r in s.routes:
            for stop in r:
                print(stop)


def main():
    x_max = 100
    y_max = 100
    station = Location(x_max / 2, y_max / 2, LocationType.station)
    customers = [
        Location(
            random.uniform(0, x_max), random.uniform(0, y_max), LocationType.customer
        )
        for _ in range(10)
    ]

    domain = Domain(station, customers)
    temperature = []
    iteration_steps = []

    start = t.time()
    best_solution = SimulatedAnnealing.main(
        domain,
        SimulatedAnnealing.cost_function,
        SimulatedAnnealing.neighbour_function,
        temperature,
        iteration_steps,
    )
    end = t.time()

    prints.solution(best_solution)
    print("Total time: ", end - start)
    # customers.insert(0, station)
    # [print(p) for p in customers]

    # Distance matrix
    # distance_matrix = [[distance(p1, p2) for p2 in customers] for p1 in customers]
    # print(distance_matrix)


if __name__ == "__main__":
    main()
