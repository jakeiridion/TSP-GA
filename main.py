import numpy as np
import random
import matplotlib.pyplot as plt
import math
from typing import List, Tuple, Union


Chromosome = List[int]
Population = List[Chromosome]
City = Tuple[int, int]
CityGrid = List[City]

MOVE_CURSOR_UP = "\033[F"
MOVE_CURSOR_DOWN = "\033[E"
CLEAR_LINE = "\r\033[2K"


class TSP:
    def __init__(self):
        self._space = (85, 115)
        # manual_city = [
        #     (55, 16), (31, 18), (49, 32), (48, 55), (60, 60), (78, 68),
        #     (53, 72), (68, 78), (57, 100), (39, 92), (36, 75), (9, 59)
        # ]
        manual_city = []
        self._city_grid = manual_city if manual_city else self._generate_city_grid(20)
        self._plot_coordinates(self._city_grid)
        self._number_of_cities = len(self._city_grid)
        self._population_size = 25
        self._tournament_size = 10
        self._crossover_probability = 0.9
        self._mutation_probability = 0.9
        self._number_of_unchanged_runs = 500
        # Line Separator Lengths
        self._runs_separator_amount = 16 + len(str(self._number_of_unchanged_runs))*2
        self.final_separator_amount = self._number_of_cities * 4 + 10

    def _generate_city_grid(self, size: int) -> CityGrid:
        coordinates = [self._generate_coordinates() for _ in range(size)]
        return coordinates

    def _generate_coordinates(self) -> City:
        max_x, max_y = self._space
        return random.randint(0, max_x), random.randint(0, max_y)

    def create_initial_population(self) -> Population:
        return [self._create_chromosome() for _ in range(self._population_size)]

    def _create_chromosome(self) -> Chromosome:
        return [0] + random.sample(list(np.arange(1, self._number_of_cities)), self._number_of_cities - 1) + [0]

    def tournament_selection(self, population: Population) -> Chromosome:
        best = random.choice(population)
        for _ in range(self._tournament_size-1):
            challenger = random.choice(population)[:]
            if self.calculate_fitness(challenger) < self.calculate_fitness(best):
                best = challenger[:]
        return best

    def calculate_fitness(self, chromosome: Chromosome) -> float:
        distance_traveled = 0
        x_current, y_current = self._city_grid[0]
        for city in chromosome:
            x_goal, y_goal = self._city_grid[city]
            distance_traveled += math.sqrt(math.pow(x_current - x_goal, 2) + math.pow(y_current - y_goal, 2))
            x_current, y_current = x_goal, y_goal
        return distance_traveled

    def cycle_crossover(self, parent1: Chromosome, parent2: Chromosome) -> Population:
        if parent1 != parent2 and self._crossover_probability > random.random():
            p1 = parent1[1:-1]
            p2 = parent2[1:-1]
            crossover_point = random.randint(0, len(p1) - 1)
            while len(set(p1)) != self._number_of_cities - 1 or p1 == parent1[1:-1]:
                if crossover_point is None:
                    break
                part_p1 = p1[crossover_point]
                part_p2 = p2[crossover_point]
                p1[crossover_point] = part_p2
                p2[crossover_point] = part_p1
                crossover_point = self._get_new_crossover_point(p1, part_p2, crossover_point)
            parent1 = [0] + p1[:] + [0]
            parent2 = [0] + p2[:] + [0]
        return [parent1, parent2]

    def _get_new_crossover_point(self, parent: Chromosome, parent_part2: int, crossover_point: int) -> Union[None, int]:
        cities = [city for city, i in enumerate(parent) if i == parent_part2]
        cities.remove(crossover_point)
        if cities:
            return cities[0]
        return None

    def mutation(self, chromosome: Chromosome) -> Chromosome:
        if self._mutation_probability > random.random():
            chromosome = chromosome[1:-1]
            to_be_replaced_index1 = random.randint(0, len(chromosome)-1)
            to_be_replaced = chromosome[to_be_replaced_index1]
            to_be_replaced_index2 = random.randint(0, len(chromosome)-1)
            chromosome[to_be_replaced_index1] = chromosome[to_be_replaced_index2]
            chromosome[to_be_replaced_index2] = to_be_replaced
            return [0] + chromosome + [0]
        return chromosome

    def run(self) -> Chromosome:
        self._setup_graph()
        population = self.create_initial_population()
        best = random.choice(population)
        unchanged_runs = 0
        runs = 0
        while unchanged_runs < self._number_of_unchanged_runs:
            # Get new best
            new_best = self._get_best_from_population(population, best)
            if new_best != best:
                unchanged_runs = 0
                best = new_best[:]
            # Selection
            population = [self.tournament_selection(population) for _ in range(self._population_size)]
            # Crossover
            crossover_population = []
            for i in range(int(self._population_size/2)):
                crossover_population += self.cycle_crossover(random.choice(population), random.choice(population))
            crossover_population = crossover_population[:] + \
                                   [best[:] for _ in range(self._population_size - len(crossover_population))]
            # Mutation
            population = [self.mutation(chromosome) for chromosome in crossover_population]
            # Update Termination Criteria
            unchanged_runs += 1
            runs += 1
            print(f"{CLEAR_LINE}" + "-" * self._runs_separator_amount)
            print(f"{CLEAR_LINE}Unchanged Run: {unchanged_runs}|{self._number_of_unchanged_runs}")
            print(f"{CLEAR_LINE}Actual Run: {runs}", end=MOVE_CURSOR_UP * 2)

        self._plot_chromosome(best)
        self._show_graph(runs)
        print(MOVE_CURSOR_DOWN * 2, end="")
        print("\n" + "-" * self.final_separator_amount)
        print(f"Starting Point: {tsp._city_grid[0]}")
        return self._get_best_from_population(population, best)

    def _get_best_from_population(self, population: Population, current_best: Chromosome) -> Chromosome:
        current_best_fitness = self.calculate_fitness(current_best)
        for chromosome in population:
            challenge_fitness = self.calculate_fitness(chromosome)
            if challenge_fitness < current_best_fitness:
                current_best = chromosome[:]
                current_best_fitness = challenge_fitness
                print(f"{CLEAR_LINE}New best: {current_best} with f: {current_best_fitness}")
        return current_best

    def _setup_graph(self):
        x, y = self._space
        plt.axis([0, x, 0, y])

    def _plot_coordinates(self, coordinates: CityGrid):
        for coordinate in coordinates:
            x, y = coordinate
            plt.plot(x, y, "ro")

    def _plot_chromosome(self, chromosome: Chromosome):
        get_next_city_coordinates = (self._city_grid[city] for city in chromosome)
        current_city_coordinates = next(get_next_city_coordinates)
        for _ in range(len(chromosome)-1):
            next_city_coordinates = next(get_next_city_coordinates)
            plt.plot([current_city_coordinates[0], next_city_coordinates[0]],
                     [current_city_coordinates[1], next_city_coordinates[1]], "b")
            current_city_coordinates = tuple(next_city_coordinates)

    def _show_graph(self, runs):
        plt.xlabel(f"Number of unchanged Iterations: {self._number_of_unchanged_runs} | "
                   f"Number of actual Iterations: {runs}")
        plt.savefig('TSP Graph.png', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    tsp = TSP()
    result = tsp.run()
    print(f"Best Individual: {result}")
    print(f"Fitness Value: {tsp.calculate_fitness(result)}")
    print("-" * tsp.final_separator_amount)
