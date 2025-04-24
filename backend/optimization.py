import numpy as np
from deap import base, creator, tools, algorithms

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, 1.0))  
creator.create("Individual", list, fitness=creator.FitnessMulti)

def evaluate(individual, weather_data, bunker_cost, vessel_data):
    route, speed = individual[0], individual[1]
    fuel_cost = speed * bunker_cost['price_per_ton']
    tce = calculate_tce(speed, route)
    time = route / speed
    return fuel_cost, time, tce

def nsga_ii_optimization(weather_data, bunker_cost, vessel_data, population_size=100, generations=50):
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 10, 20)  
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 2)  
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, weather_data=weather_data, bunker_cost=bunker_cost, vessel_data=vessel_data)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    
    population = toolbox.population(n=population_size)
    algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=population_size, cxpb=0.6, mutpb=0.3, ngen=generations, verbose=True)
    
    return population

def topsis_solution_ranking(pareto_solutions):
    normalized_solutions = pareto_solutions / np.linalg.norm(pareto_solutions, axis=0)
    ideal_solution = np.max(normalized_solutions, axis=0)
    anti_ideal_solution = np.min(normalized_solutions, axis=0)
    distances_to_ideal = np.linalg.norm(normalized_solutions - ideal_solution, axis=1)
    distances_to_anti_ideal = np.linalg.norm(normalized_solutions - anti_ideal_solution, axis=1)
    closeness_coefficients = distances_to_anti_ideal / (distances_to_ideal + distances_to_anti_ideal)
    best_solution_index = np.argmax(closeness_coefficients)
    return pareto_solutions[best_solution_index]
