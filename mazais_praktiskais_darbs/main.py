import math
import random
from dataclasses import dataclass
from typing import List, Callable, Dict, Tuple
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

    def __str__(self):
        return f"Cost(hard={self.hard}, soft={self.soft})"


class LocationType(Enum):
    customer = "customer"
    station = "station"


@dataclass
class Location(Point):
    id: int
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


@dataclass
class TemperatureStep:
    temperature: float
    iteration_steps_constant: int


@dataclass
class CoolingSchedule:
    customer_count: int
    temperature_steps: List[TemperatureStep]

    def get_iteration_steps(self) -> List[int]:
        return [
            int(int(step.iteration_steps_constant**1.5) * self.customer_count)
            for step in self.temperature_steps
        ]

    def get_temperatures(self) -> List[float]:
        return [step.temperature for step in self.temperature_steps]


class SimulatedAnnealing:

    @staticmethod
    def main(
        domain: Domain,
        cost_function: Callable[[Solution], Cost],
        neighbour_function: Callable[[Solution], Solution],
        temperature: List[float],  # t[k]
        iteration_steps: List[int],  # L[k]
    ) -> Tuple[Solution, Solution]:
        initial_solution = SimulatedAnnealing.initialize_solution(domain)
        current_solution = initial_solution
        best_solution = initial_solution
        current_cost = cost_function(current_solution).soft
        best_cost = current_cost

        # 3. Lekcijas 7. slaida algoritms
        k = 0

        while k < len(temperature):
            t_k = temperature[k]
            L_k = iteration_steps[k]

            for _ in range(L_k):

                neighbor_solution = neighbour_function(current_solution)
                neighbor_cost = cost_function(neighbor_solution).soft

                delta = neighbor_cost - current_cost

                # Akceptēšanas kritērijs
                if delta < 0 or random.random() < math.exp(-delta / t_k):
                    current_solution = neighbor_solution
                    current_cost = neighbor_cost

                    if current_cost < best_cost:
                        best_solution = current_solution
                        best_cost = current_cost

            k += 1

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
    def neighbour_function(solution: Solution) -> Solution:
        neighbor_solution = Solution(routes=[])
        for route in solution.routes:
            new_route = Route(route=route.route[:])
            neighbor_solution.routes.append(new_route)

        # Pagaidām tiek uzskatīts, ka ir tikai 1 maršruts katram risinājumam, bet to šeit var pielāgot
        route = neighbor_solution.routes[0].route

        # Izvēlamies 2 nejaušus indeksus, ko mainīt vietām, izņemot sākuma un beigu punktus (jo tā ir stacija, kura ir nemainīga)
        idx1, idx2 = random.sample(range(1, len(route) - 1), 2)
        # Apmainām vietām šos 2 klientus
        route[idx1], route[idx2] = route[idx2], route[idx1]

        return neighbor_solution


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
            for location in route.route:
                ax.annotate(location.name.value[:4].upper(), (location.x, location.y))
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
    customer_count = 50
    station = Location(x_max / 2, y_max / 2, 1, LocationType.station)
    customers = [
        Location(
            random.uniform(0, x_max),
            random.uniform(0, y_max),
            i + 2,
            LocationType.customer,
        )
        for i in range(customer_count)
    ]

    domain = Domain(station, customers)

    cooling_schedule = CoolingSchedule(
        customer_count,
        temperature_steps=[
            TemperatureStep(temperature=1000, iteration_steps_constant=100),
            TemperatureStep(temperature=900, iteration_steps_constant=100),
            TemperatureStep(temperature=800, iteration_steps_constant=100),
            TemperatureStep(temperature=700, iteration_steps_constant=100),
            TemperatureStep(temperature=600, iteration_steps_constant=100),
            TemperatureStep(temperature=500, iteration_steps_constant=100),
            TemperatureStep(temperature=400, iteration_steps_constant=80),
            TemperatureStep(temperature=300, iteration_steps_constant=80),
            TemperatureStep(temperature=200, iteration_steps_constant=60),
            TemperatureStep(temperature=100, iteration_steps_constant=60),
            TemperatureStep(temperature=50, iteration_steps_constant=40),
            TemperatureStep(temperature=25, iteration_steps_constant=40),
            TemperatureStep(temperature=10, iteration_steps_constant=30),
            TemperatureStep(temperature=5, iteration_steps_constant=20),
            TemperatureStep(temperature=1, iteration_steps_constant=10),
        ],
    )

    start = t.time()
    initial_solution, best_solution = SimulatedAnnealing.main(
        domain,
        SimulatedAnnealing.cost_function,
        SimulatedAnnealing.neighbour_function,
        cooling_schedule.get_temperatures(),
        cooling_schedule.get_iteration_steps(),
    )
    end = t.time()

    prints.solution(best_solution)
    print("Total time: ", end - start)

    plot_main(initial_solution, best_solution)


if __name__ == "__main__":
    main()
