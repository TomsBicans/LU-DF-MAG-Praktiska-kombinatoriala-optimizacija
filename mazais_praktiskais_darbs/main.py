import math
import random
from dataclasses import dataclass
from typing import List, Callable, Dict
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
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

    # To string method
    def __str__(self):
        return f"Cost(hard={self.hard}, soft={self.soft})"


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
        cost_function: Callable[[Solution], Cost],
        neighbour_function: Callable[[Solution], List[Solution]],
        temperature: List[float],  # TODO: check if type is correct
        iteration_steps: List[int],  # TODO: check if type is correct
    ) -> Dict[Solution, Solution]:
        initial_solution = SimulatedAnnealing.initialize_solution(domain)
        best_solution = initial_solution
        # TODO: implement the rest of the algorithm. Can be found in L3 slide 7
        return initial_solution, best_solution

    @staticmethod
    def initialize_solution(domain: Domain) -> Solution:
        customers = domain.customers[:]
        random.shuffle(customers)
        route = [domain.station] + customers + [domain.station]
        return Solution(routes=[Route(route=route)])

    @staticmethod
    def cost_function(solution: Solution) -> Cost:
        # Rēķinām kopējo distanci starp maršruta punktiem,
        # bet var papildināt domēnu un rēķināt arī enerģijas patēriņu.
        def distance_cost(solution: Solution) -> Cost:
            total_distance = 0.0
            for route in solution.routes:
                route_distance = 0.0
                locations = route.route
                for i in range(len(locations) - 1):
                    route_distance += distance(locations[i], locations[i + 1])
                total_distance += route_distance

            return Cost(0.0, total_distance)

        return distance_cost(solution)

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
            for stop in r.route:
                print(stop)


def plot_main(initial_solution: Solution, best_solution: Solution):
    def plot_solution(ax: Axes, solution: Solution, title: str):
        for route in solution.routes:
            x = [location.x for location in route.route]
            y = [location.y for location in route.route]
            ax.plot(x, y, marker="o")
            # Annotate points
            for location in route.route:
                ax.annotate(location.name.value[0].upper(), (location.x, location.y))
        ax.set_title(title)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.grid(True)

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    # Sākotnējais risinājums
    plot_solution(
        axs[0],
        initial_solution,
        f"Initial Solution\n {SimulatedAnnealing.cost_function(initial_solution)}",
    )

    # Labākais atrastais risinājums
    plot_solution(
        axs[1],
        best_solution,
        f"Optimized Solution\n {SimulatedAnnealing.cost_function(best_solution)}",
    )

    plt.tight_layout()
    plt.show()


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
    initial_solution, best_solution = SimulatedAnnealing.main(
        domain,
        SimulatedAnnealing.cost_function,
        SimulatedAnnealing.neighbour_function,
        temperature,
        iteration_steps,
    )
    end = t.time()

    prints.solution(best_solution)
    print("Total time: ", end - start)

    plot_main(initial_solution, best_solution)


if __name__ == "__main__":
    main()
