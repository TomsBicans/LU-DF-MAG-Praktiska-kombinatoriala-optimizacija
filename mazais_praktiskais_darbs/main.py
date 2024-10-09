import math
import random
from dataclasses import dataclass
from typing import List, Callable, Tuple
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from enum import Enum
import time as t
from tqdm import tqdm
import os
import pandas as pd


SHOW_CHARTS = False


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
    name: str


@dataclass
class Route:
    route: List[Location]


@dataclass
class Domain:
    station: Location
    customers: List[Location]


@dataclass
class Solution:
    routes: List[Route]

    def __repr__(self):
        return f"Solution(routes={self.routes})"

    def serialize(self, file_path: str) -> None:
        """
        Serializes the Solution instance to a text file using the __repr__ method.
        """
        with open(file_path, "w") as file:
            file.write(repr(self))

    @staticmethod
    def deserialize(file_path: str) -> "Solution":
        """
        Deserializes the Solution instance from a text file using eval().
        """
        with open(file_path, "r") as file:
            data = file.read()
            return eval(data)


@dataclass
class TemperatureStep:
    temperature: float
    iteration_steps_base: int


@dataclass
class CoolingSchedule:
    customer_count: int
    temperature_steps: List[TemperatureStep]

    def get_iteration_steps(self) -> List[int]:
        return [
            int(int(step.iteration_steps_base**2) * self.customer_count)
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

        for k in tqdm(range(len(temperature)), desc="Temperature steps"):
            t_k = temperature[k]
            L_k = iteration_steps[k]

            for _ in tqdm(range(L_k), desc=f"Iteration steps at T={t_k}"):

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
            def distance(p1: Point, p2: Point) -> float:
                return math.hypot(p1.x - p2.x, p1.y - p2.y)

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


def plot_main(
    initial_solution: Solution,
    best_solution: Solution,
    store_path: str,
    customer_count: int,
    total_time: float,
    test_name: str,
    domain_variation_name: str,
):

    def plot_solution(ax: Axes, solution: Solution, title: str):
        for route in solution.routes:
            x = [location.x for location in route.route]
            y = [location.y for location in route.route]
            ax.plot(x, y, marker="o")
            for location in route.route:
                ax.annotate(location.name[:1].upper(), (location.x, location.y))
        ax.set_title(title)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.grid(True)

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    # Sākotnējais risinājums
    plot_solution(
        axs[0],
        initial_solution,
        f"Scenario: {test_name}\n Domain variation name: {domain_variation_name}\n Initial Solution\n {SimulatedAnnealing.cost_function(initial_solution)}\n Customer Count: {customer_count}",
    )

    # Labākais atrastais risinājums
    plot_solution(
        axs[1],
        best_solution,
        f"Scenario: {test_name}\n Domain variation name: {domain_variation_name}\n Optimized Solution\n {SimulatedAnnealing.cost_function(best_solution)}\n Customer Count: {customer_count}\n Total calculation time: {round(total_time, 2)}s",
    )

    plt.tight_layout()
    plt.savefig(store_path)
    if SHOW_CHARTS:
        plt.show()


def plot_performance(df: pd.DataFrame, store_path: str):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Plot execution time vs. customer count
    axs[0].plot(df["customer_count"], df["total_time"], marker="o")
    axs[0].set_xlabel("Number of Customers")
    axs[0].set_ylabel("Execution Time (s)")
    axs[0].set_title("Execution Time vs. Number of Customers")
    axs[0].grid(True)

    # Plot cost reduction vs. customer count
    cost_reduction = df["initial_cost"] - df["best_cost"]
    axs[1].plot(df["customer_count"], cost_reduction, marker="o", color="green")
    axs[1].set_xlabel("Number of Customers")
    axs[1].set_ylabel("Cost Reduction")
    axs[1].set_title("Cost Reduction vs. Number of Customers")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(store_path)
    if SHOW_CHARTS:
        plt.show()


@dataclass
class TestCase:
    name: str
    domain_variations: List[Callable[[], Domain]]
    customer_count: int
    x_max: int
    y_max: int
    test_directory: str


class ExecutionMode(Enum):
    LOAD = 1
    GENERATE = 2


def file_write(path: str, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)


def file_read(path: str) -> str:
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    else:
        return ""


class Examples:
    def basic_domain(test_case: TestCase) -> Domain:
        station = Location(
            test_case.x_max / 2,
            test_case.y_max / 2,
            1,
            LocationType.station.value,
        )
        customers = [
            Location(
                random.uniform(0, test_case.x_max),
                random.uniform(0, test_case.y_max),
                i + 2,
                LocationType.customer.value,
            )
            for i in range(test_case.customer_count)
        ]

        domain = Domain(station, customers)
        return domain

    def three_cities(test_case: TestCase) -> Domain:
        cluster_centers = [
            (test_case.x_max * 0.25, test_case.y_max * 0.25),
            (test_case.x_max * 0.75, test_case.y_max * 0.25),
            (test_case.x_max * 0.5, test_case.y_max * 0.75),
        ]

        cluster_radius = min(test_case.x_max, test_case.y_max) * 0.1

        customers_per_cluster = test_case.customer_count // 3
        remainder = test_case.customer_count % 3

        customers = []
        customer_id = 2

        for i, (center_x, center_y) in enumerate(cluster_centers):
            count = customers_per_cluster + (1 if i < remainder else 0)

            for _ in range(count):

                angle = random.uniform(0, 2 * 3.14159)
                radius = random.uniform(0, cluster_radius)
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)

                customers.append(
                    Location(x, y, customer_id, LocationType.customer.value)
                )
                customer_id += 1

        station = Location(
            test_case.x_max / 2, test_case.y_max / 2, 1, LocationType.station.value
        )

        domain = Domain(station, customers)
        return domain

    def seven_cities(test_case: TestCase) -> Domain:
        cluster_centers = [
            (test_case.x_max * 0.2, test_case.y_max * 0.2),
            (test_case.x_max * 0.8, test_case.y_max * 0.2),
            (test_case.x_max * 0.5, test_case.y_max * 0.35),
            (test_case.x_max * 0.2, test_case.y_max * 0.8),
            (test_case.x_max * 0.8, test_case.y_max * 0.8),
            (test_case.x_max * 0.35, test_case.y_max * 0.55),
            (test_case.x_max * 0.65, test_case.y_max * 0.55),
        ]

        cluster_radius = min(test_case.x_max, test_case.y_max) * 0.1

        customers_per_cluster = test_case.customer_count // 7
        remainder = test_case.customer_count % 7

        customers = []
        customer_id = 2

        for i, (center_x, center_y) in enumerate(cluster_centers):

            count = customers_per_cluster + (1 if i < remainder else 0)

            for _ in range(count):
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0, cluster_radius)
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)

                x = max(0, min(test_case.x_max, x))
                y = max(0, min(test_case.y_max, y))

                customers.append(
                    Location(x, y, customer_id, LocationType.customer.value)
                )
                customer_id += 1

        station = Location(
            test_case.x_max / 2,
            test_case.y_max / 2,
            1,
            LocationType.station.value,
        )

        domain = Domain(station, customers)
        return domain


def main():
    execution_mode = ExecutionMode.LOAD

    test_cases: List[TestCase] = [
        TestCase(
            "Small",
            [Examples.basic_domain, Examples.three_cities, Examples.seven_cities],
            10,
            100,
            100,
            "tests/1/",
        ),
        TestCase(
            "Medium",
            [Examples.basic_domain, Examples.three_cities, Examples.seven_cities],
            20,
            100,
            100,
            "tests/2/",
        ),
        TestCase(
            "Large",
            [Examples.basic_domain, Examples.three_cities, Examples.seven_cities],
            30,
            100,
            100,
            "tests/3/",
        ),
        TestCase(
            "Huge",
            [Examples.basic_domain, Examples.three_cities, Examples.seven_cities],
            40,
            100,
            100,
            "tests/4/",
        ),
        TestCase(
            "Giant",
            [Examples.basic_domain, Examples.three_cities, Examples.seven_cities],
            50,
            100,
            100,
            "tests/5/",
        ),
        TestCase(
            "Colossal",
            [Examples.basic_domain, Examples.three_cities, Examples.seven_cities],
            100,
            100,
            100,
            "tests/6/",
        ),
        TestCase(
            "Gigantic",
            [Examples.basic_domain, Examples.three_cities, Examples.seven_cities],
            200,
            100,
            100,
            "tests/7/",
        ),
    ]

    results = []

    for test_case in test_cases:
        test_directory = test_case.test_directory
        if not os.path.exists(test_directory):
            os.makedirs(test_directory)

        for domain_variation in test_case.domain_variations:

            subname = domain_variation.__name__
            location_initial_location = os.path.join(
                test_case.test_directory, f"initial_solution_{subname}.txt"
            )
            location_result_location = os.path.join(
                test_case.test_directory, f"best_solution_{subname}.txt"
            )
            execution_time_save_location = os.path.join(
                test_case.test_directory, f"execution_time_{subname}.txt"
            )

            if execution_mode == ExecutionMode.GENERATE or not all(
                [
                    os.path.exists(location_initial_location),
                    os.path.exists(location_result_location),
                    os.path.exists(execution_time_save_location),
                ]
            ):

                domain = domain_variation(test_case)

                cooling_schedule = CoolingSchedule(
                    test_case.customer_count,
                    temperature_steps=[
                        TemperatureStep(temperature=1000, iteration_steps_base=50),
                        TemperatureStep(temperature=900, iteration_steps_base=50),
                        TemperatureStep(temperature=800, iteration_steps_base=50),
                        TemperatureStep(temperature=700, iteration_steps_base=50),
                        TemperatureStep(temperature=600, iteration_steps_base=50),
                        TemperatureStep(temperature=500, iteration_steps_base=50),
                        TemperatureStep(temperature=400, iteration_steps_base=50),
                        TemperatureStep(temperature=300, iteration_steps_base=50),
                        TemperatureStep(temperature=200, iteration_steps_base=50),
                        TemperatureStep(temperature=100, iteration_steps_base=50),
                        TemperatureStep(temperature=50, iteration_steps_base=50),
                        TemperatureStep(temperature=25, iteration_steps_base=50),
                        TemperatureStep(temperature=10, iteration_steps_base=50),
                        TemperatureStep(temperature=5, iteration_steps_base=50),
                        TemperatureStep(temperature=1, iteration_steps_base=50),
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

                best_solution.serialize(location_result_location)
                initial_solution.serialize(location_initial_location)
                total_time = str(end - start)
                file_write(execution_time_save_location, total_time)
                prints.solution(best_solution)
                print("Total time: ", end - start)

            elif execution_mode == ExecutionMode.LOAD:
                initial_solution = Solution.deserialize(location_initial_location)
                best_solution = Solution.deserialize(location_result_location)
                total_time = (
                    float(file_read(execution_time_save_location).strip())
                    if file_read(execution_time_save_location)
                    else 0
                )

            results.append(
                {
                    "customer_count": test_case.customer_count,
                    "initial_cost": SimulatedAnnealing.cost_function(
                        initial_solution
                    ).soft,
                    "best_cost": SimulatedAnnealing.cost_function(best_solution).soft,
                    "total_time": total_time,
                }
            )

            plot_save_location = os.path.join(
                test_case.test_directory, f"plot_{subname}.png"
            )
            plot_main(
                initial_solution,
                best_solution,
                plot_save_location,
                test_case.customer_count,
                total_time,
                test_case.name,
                domain_variation.__name__,
            )

    plot_performance(pd.DataFrame(results), "tests/overview.png")


if __name__ == "__main__":
    main()
