"""Genetic algorithm simulation for optimizing building configurations."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simtools.models.building import Building
    from simtools.models.resource import Resource


@dataclass
class BuildingGene:
    """Represents a single building in the DNA.
    
    Attributes:
        building_name: Name of the building type.
        level: Building level (1 to SimulationConfig.max_level).
    """
    building_name: str
    level: int = 1
    
    def copy(self) -> "BuildingGene":
        """Create a copy of this gene."""
        return BuildingGene(building_name=self.building_name, level=self.level)


@dataclass
class Individual:
    """Represents an individual in the genetic algorithm population.
    
    The DNA is a list of BuildingGene objects representing the buildings
    owned by the company.
    """
    genes: list[BuildingGene] = field(default_factory=list)
    fitness: float = 0.0
    total_cost: float = 0.0
    
    def copy(self) -> "Individual":
        """Create a deep copy of this individual."""
        return Individual(
            genes=[g.copy() for g in self.genes],
            fitness=self.fitness,
            total_cost=self.total_cost,
        )


@dataclass
class SimulationConfig:
    """Configuration for the genetic algorithm simulation.
    
    Attributes:
        slots: Number of building slots available.
        budget: Maximum investment budget for buildings.
        population_size: Size of the population.
        generations: Number of generations to run.
        mutation_rate: Probability of mutation (0.0-1.0).
        crossover_rate: Probability of crossover (0.0-1.0).
        max_level: Maximum building level allowed.
        elitism: Number of best individuals to preserve each generation.
        tournament_size: Size of tournament for selection.
        budget_penalty_factor: Multiplier for budget overage penalty.
    """
    slots: int = 5
    budget: float = 100000.0
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    max_level: int = 10
    elitism: int = 2
    tournament_size: int = 3
    budget_penalty_factor: float = 2.0


class GeneticAlgorithm:
    """Genetic algorithm for optimizing building configurations.
    
    This class implements a genetic algorithm to find the optimal set of
    buildings and their levels to maximize profit over a 48-hour simulation.
    """
    
    def __init__(
        self,
        config: SimulationConfig,
        buildings: list[Building],
        resources: list[Resource],
        price_map: dict[int, float],
        q0_price_map: dict[int, float],
        transport_price: float,
        name_to_id: dict[str, int],
        abundance: float = 90.0,
        admin_overhead: float = 0.0,
        has_robots: bool = False,
    ):
        """Initialize the genetic algorithm.
        
        Args:
            config: Simulation configuration.
            buildings: List of available building types.
            resources: List of all resources.
            price_map: Map of resource ID to price at target quality.
            q0_price_map: Map of resource ID to Q0 price for building costs.
            transport_price: Price per transport unit.
            name_to_id: Map of resource name (lowercase) to resource ID.
            abundance: Abundance percentage for mine/well resources.
            admin_overhead: Administrative overhead percentage.
            has_robots: Whether robots are used (3% wage reduction).
        """
        self.config = config
        self.buildings = buildings
        self.resources = resources
        self.price_map = price_map
        self.q0_price_map = q0_price_map
        self.transport_price = transport_price
        self.name_to_id = name_to_id
        self.abundance = abundance
        self.admin_overhead = admin_overhead
        self.has_robots = has_robots
        
        # Build lookup maps
        self.building_by_name = {b.name: b for b in buildings}
        self.resource_by_name = {r.name.lower(): r for r in resources}
        self.resource_by_id = {r.id: r for r in resources}
        
        # Build building to resources map
        self.building_to_resources: dict[str, list[Resource]] = {}
        for building in buildings:
            resources_list = []
            for res_name in building.produces:
                res = self.resource_by_name.get(res_name.lower())
                if res:
                    resources_list.append(res)
            if resources_list:
                self.building_to_resources[building.name] = resources_list
        
        # Filter to buildings that produce resources with prices
        self.valid_buildings = [
            b for b in buildings
            if b.name in self.building_to_resources
            and any(
                self.price_map.get(r.id, 0) > 0
                for r in self.building_to_resources[b.name]
            )
        ]
        
        # Track best fitness over generations
        self.fitness_history: list[float] = []
    
    def calculate_building_cost(self, building_name: str, level: int) -> float:
        """Calculate the total cost of a building at a given level.
        
        Args:
            building_name: Name of the building.
            level: Target level.
            
        Returns:
            Total cost including construction and upgrades.
        """
        building = self.building_by_name.get(building_name)
        if not building:
            return 0.0
        
        # Base construction cost
        base_cost, _ = building.calculate_construction_cost(
            self.q0_price_map, self.name_to_id
        )
        
        # Add upgrade costs (sum of k * base_cost for k = 1 to level-1)
        # This is consistent with Building.calculate_upgrade_cost which uses:
        # step cost from k to k+1 is k * base_cost
        # So total upgrade from 1 to level = sum(k for k=1 to level-1) = level*(level-1)/2
        if level <= 1:
            return base_cost
        
        # Total upgrade cost = base_cost * (1 + 2 + ... + (level-1)) = base_cost * level * (level-1) / 2
        upgrade_cost = base_cost * level * (level - 1) / 2
        return base_cost + upgrade_cost
    
    def calculate_individual_cost(self, individual: Individual) -> float:
        """Calculate the total cost of all buildings for an individual.
        
        Args:
            individual: The individual to calculate cost for.
            
        Returns:
            Total investment cost.
        """
        total = 0.0
        for gene in individual.genes:
            total += self.calculate_building_cost(gene.building_name, gene.level)
        return total
    
    def get_best_resource_for_building(self, building_name: str) -> Resource | None:
        """Get the most profitable resource for a building.
        
        Args:
            building_name: Name of the building.
            
        Returns:
            Most profitable resource or None if no valid resources.
        """
        resources = self.building_to_resources.get(building_name, [])
        if not resources:
            return None
        
        best_resource = None
        best_profit = float("-inf")
        
        for res in resources:
            selling_price = self.price_map.get(res.id, 0)
            if selling_price <= 0:
                continue
            
            profit_data = res.calculate_profit(
                selling_price=selling_price,
                input_prices=self.price_map,
                transport_price=self.transport_price,
                abundance=self.abundance,
                admin_overhead=self.admin_overhead,
                is_contract=False,
                has_robots=self.has_robots,
            )
            
            if profit_data["profit_per_hour"] > best_profit:
                best_profit = profit_data["profit_per_hour"]
                best_resource = res
        
        return best_resource
    
    def simulate_48_hours(self, individual: Individual) -> float:
        """Simulate 48 hours of production and calculate profit.
        
        Rules:
        - Each hour, buildings produce resources
        - Production first uses company inventory for inputs
        - Missing inputs are bought from market
        - At end, all inventory is sold (with fees and transport costs)
        - Net profit = sales - purchases
        
        Args:
            individual: The individual to simulate.
            
        Returns:
            Net profit from the 48-hour simulation.
        """
        if not individual.genes:
            return 0.0
        
        # Company inventory (resource_id -> quantity)
        inventory: dict[int, float] = {}
        # Total money spent on market purchases
        total_purchases = 0.0
        
        # Determine what each building produces
        building_production: list[tuple[Resource, int]] = []
        for gene in individual.genes:
            resource = self.get_best_resource_for_building(gene.building_name)
            if resource:
                building_production.append((resource, gene.level))
        
        if not building_production:
            return 0.0
        
        # Simulate 48 hours
        for _ in range(48):
            for resource, level in building_production:
                # Calculate production per hour at this level
                produced = resource.get_effective_production(self.abundance) * level
                
                # Calculate input requirements
                for input_id, input_info in resource.inputs.items():
                    required = input_info.quantity * produced
                    
                    # Use inventory first
                    available = inventory.get(input_id, 0.0)
                    if available >= required:
                        inventory[input_id] = available - required
                    else:
                        # Buy the rest from market
                        to_buy = required - available
                        inventory[input_id] = 0.0
                        
                        price = self.price_map.get(input_id, 0)
                        total_purchases += price * to_buy
                
                # Add produced resource to inventory
                inventory[resource.id] = inventory.get(resource.id, 0.0) + produced
        
        # Sell all inventory at end
        total_sales = 0.0
        for res_id, quantity in inventory.items():
            if quantity <= 0:
                continue
            
            selling_price = self.price_map.get(res_id, 0)
            if selling_price <= 0:
                continue
            
            # Gross revenue
            revenue = selling_price * quantity
            
            # Apply 4% market fee
            market_fee = revenue * 0.04
            
            # Calculate transport cost
            resource = self.resource_by_id.get(res_id)
            if resource:
                transport_cost = resource.transportation * self.transport_price * quantity
            else:
                transport_cost = 0.0
            
            total_sales += revenue - market_fee - transport_cost
        
        # Net profit
        return total_sales - total_purchases
    
    def evaluate_fitness(self, individual: Individual) -> float:
        """Evaluate the fitness of an individual.
        
        Args:
            individual: The individual to evaluate.
            
        Returns:
            Fitness value (profit with budget penalty).
        """
        # Calculate total cost
        total_cost = self.calculate_individual_cost(individual)
        individual.total_cost = total_cost
        
        # Simulate production
        profit = self.simulate_48_hours(individual)
        
        # Apply penalty if over budget
        # Use a ratio-based penalty that scales with how much over budget we are
        if total_cost > self.config.budget:
            overage_ratio = total_cost / self.config.budget
            # The penalty grows quadratically with overage ratio
            # Use the budget as the base for penalty calculation to ensure consistent scaling
            # If 2x over budget, penalty = 4 * budget * penalty_factor
            # This ensures over-budget configurations are strongly discouraged
            penalty = overage_ratio ** 2 * self.config.budget * self.config.budget_penalty_factor
            profit -= penalty
        
        individual.fitness = profit
        return profit
    
    def create_random_individual(self) -> Individual:
        """Create a random individual within constraints.
        
        Returns:
            A new random individual.
        """
        individual = Individual()
        
        if not self.valid_buildings:
            return individual
        
        # Randomly determine number of buildings (1 to slots)
        num_buildings = random.randint(1, self.config.slots)
        
        for _ in range(num_buildings):
            building = random.choice(self.valid_buildings)
            level = random.randint(1, self.config.max_level)
            individual.genes.append(BuildingGene(building.name, level))
        
        return individual
    
    def initialize_population(self) -> list[Individual]:
        """Initialize the population with random individuals.
        
        Returns:
            List of random individuals.
        """
        return [
            self.create_random_individual()
            for _ in range(self.config.population_size)
        ]
    
    def tournament_select(self, population: list[Individual]) -> Individual:
        """Select an individual using tournament selection.
        
        Args:
            population: Current population.
            
        Returns:
            Selected individual.
        """
        tournament = random.sample(
            population,
            min(self.config.tournament_size, len(population))
        )
        return max(tournament, key=lambda ind: ind.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        """Perform crossover between two parents.
        
        Uses a building-aware crossover that maintains valid slot counts:
        - Randomly select a crossover point for each parent
        - Combine genes from both parents
        - Trim to max slots if needed
        
        Args:
            parent1: First parent.
            parent2: Second parent.
            
        Returns:
            Two offspring individuals.
        """
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        if not parent1.genes or not parent2.genes:
            return parent1.copy(), parent2.copy()
        
        # Single-point crossover with gene lists
        point1 = random.randint(0, len(parent1.genes))
        point2 = random.randint(0, len(parent2.genes))
        
        # Create offspring
        child1_genes = [g.copy() for g in parent1.genes[:point1]] + [g.copy() for g in parent2.genes[point2:]]
        child2_genes = [g.copy() for g in parent2.genes[:point2]] + [g.copy() for g in parent1.genes[point1:]]
        
        # Trim to max slots
        child1_genes = child1_genes[:self.config.slots]
        child2_genes = child2_genes[:self.config.slots]
        
        return Individual(genes=child1_genes), Individual(genes=child2_genes)
    
    def mutate(self, individual: Individual) -> Individual:
        """Mutate an individual.
        
        Possible mutations:
        - Change building level
        - Swap building type
        - Add a building (if under slot limit)
        - Remove a building (if multiple buildings)
        
        Args:
            individual: Individual to mutate.
            
        Returns:
            Mutated individual (may be same if no mutation).
        """
        if random.random() > self.config.mutation_rate:
            return individual
        
        if not individual.genes:
            # Add a random building if empty
            if self.valid_buildings:
                building = random.choice(self.valid_buildings)
                level = random.randint(1, self.config.max_level)
                individual.genes.append(BuildingGene(building.name, level))
            return individual
        
        # Choose mutation type
        mutation_type = random.choice(["level", "swap", "add", "remove"])
        
        if mutation_type == "level":
            # Change level of random gene
            gene_idx = random.randint(0, len(individual.genes) - 1)
            # Adjust level by -2 to +2, keeping in bounds
            delta = random.randint(-2, 2)
            new_level = max(1, min(self.config.max_level, individual.genes[gene_idx].level + delta))
            individual.genes[gene_idx].level = new_level
        
        elif mutation_type == "swap":
            # Swap building type
            if self.valid_buildings:
                gene_idx = random.randint(0, len(individual.genes) - 1)
                building = random.choice(self.valid_buildings)
                individual.genes[gene_idx].building_name = building.name
        
        elif mutation_type == "add":
            # Add a building if under slot limit
            if len(individual.genes) < self.config.slots and self.valid_buildings:
                building = random.choice(self.valid_buildings)
                level = random.randint(1, self.config.max_level)
                individual.genes.append(BuildingGene(building.name, level))
        
        elif mutation_type == "remove":
            # Remove a building if more than one
            if len(individual.genes) > 1:
                gene_idx = random.randint(0, len(individual.genes) - 1)
                individual.genes.pop(gene_idx)
        
        return individual
    
    def run(self, progress_callback=None) -> tuple[Individual, list[float]]:
        """Run the genetic algorithm.
        
        Args:
            progress_callback: Optional callback function(generation, best_fitness, avg_fitness)
                              called each generation for progress updates.
        
        Returns:
            Tuple of (best individual, fitness history).
        """
        # Initialize population
        population = self.initialize_population()
        
        # Evaluate initial fitness
        for ind in population:
            self.evaluate_fitness(ind)
        
        self.fitness_history = []
        
        for generation in range(self.config.generations):
            # Sort by fitness
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            
            # Track best fitness
            best_fitness = population[0].fitness
            avg_fitness = sum(ind.fitness for ind in population) / len(population)
            self.fitness_history.append(best_fitness)
            
            # Call progress callback
            if progress_callback:
                progress_callback(generation + 1, best_fitness, avg_fitness)
            
            # Create new population
            new_population = []
            
            # Elitism - keep best individuals
            for i in range(min(self.config.elitism, len(population))):
                new_population.append(population[i].copy())
            
            # Generate rest of population through selection, crossover, mutation
            while len(new_population) < self.config.population_size:
                # Selection
                parent1 = self.tournament_select(population)
                parent2 = self.tournament_select(population)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Evaluate fitness
                self.evaluate_fitness(child1)
                self.evaluate_fitness(child2)
                
                new_population.append(child1)
                if len(new_population) < self.config.population_size:
                    new_population.append(child2)
            
            population = new_population
        
        # Final sort and return best
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        return population[0], self.fitness_history


def render_ascii_graph(data: list[float], width: int = 60, height: int = 15) -> list[str]:
    """Render an ASCII graph of fitness over generations.
    
    Args:
        data: List of fitness values.
        width: Width of the graph in characters.
        height: Height of the graph in lines.
        
    Returns:
        List of strings representing the graph lines.
    """
    if not data:
        return ["No data to display"]
    
    # Handle single value or all same values
    if len(data) == 1:
        return [f"Single data point: {data[0]:,.2f}"]
    
    # Normalize data to fit in height
    min_val = min(data)
    max_val = max(data)
    
    # Handle case where all values are the same
    if max_val == min_val:
        # Display a flat line at middle height
        graph = [[" " for _ in range(min(width, len(data)))] for _ in range(height)]
        mid_row = height // 2
        for x in range(min(width, len(data))):
            graph[mid_row][x] = "█"
        
        lines = []
        for i, row in enumerate(graph):
            if i == 0 or i == height - 1 or i == height // 2:
                label = f"{max_val:>12,.0f} │"
            else:
                label = "             │"
            lines.append(label + "".join(row))
        
        lines.append("             └" + "─" * len(graph[0]))
        lines.append(f"              1 (constant value){' ' * (len(graph[0]) - 20)}{len(data)}")
        return lines
    
    value_range = max_val - min_val
    
    # Create empty graph
    graph = [[" " for _ in range(width)] for _ in range(height)]
    
    # Sample data points to fit width - use at least width points or all data if less
    if len(data) <= width:
        sampled = data
    else:
        step = len(data) / width
        sampled = [data[int(i * step)] for i in range(width)]
    
    # Plot points with area fill
    for x, value in enumerate(sampled):
        if x >= width:
            break
        # Normalize to height
        y = int((value - min_val) / value_range * (height - 1))
        y = max(0, min(height - 1, y))
        # Fill from bottom to this point
        for fill_y in range(y + 1):
            row_idx = height - 1 - fill_y
            if fill_y == y:
                graph[row_idx][x] = "█"  # Top of the bar
            else:
                graph[row_idx][x] = "▒"  # Fill under the line
    
    # Add axis labels
    lines = []
    for i, row in enumerate(graph):
        # Add y-axis label for first, middle, and last rows
        if i == 0:
            label = f"{max_val:>12,.0f} │"
        elif i == height - 1:
            label = f"{min_val:>12,.0f} │"
        elif i == height // 2:
            mid_val = (max_val + min_val) / 2
            label = f"{mid_val:>12,.0f} │"
        else:
            label = "             │"
        lines.append(label + "".join(row))
    
    # Add x-axis
    lines.append("             └" + "─" * width)
    
    # Format x-axis label
    end_label = str(len(data))
    padding_needed = width - len(end_label) - len("Generation") - 1
    left_pad = padding_needed // 2
    right_pad = padding_needed - left_pad
    lines.append(f"              1{' ' * left_pad}Generation{' ' * right_pad}{end_label}")
    
    return lines
