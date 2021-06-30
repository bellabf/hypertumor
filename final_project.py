#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 07:48:17 2020

@author: isabella
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class Chromosome:

    def __init__(self, overpop, underpop, reproduction, mutation_rate):
        self.overpop = overpop
        self.underpop = underpop
        self.reproduction = reproduction
        self.mutation_rate = mutation_rate


    def getoverpop(self):
        return self.overpop

    def getunderpop(self):
        return self.underpop

    def getreproduction(self):
        return self.reproduction

    def getmutationrate(self):
        return self.mutation_rate


def number_neighbours(i, j, grid):
    number_alive = grid[i - 1][j - 1] + grid[i - 1][j] + grid[i - 1][j + 1]  # before row  neighbours
    number_alive += grid[i][j - 1] + grid[i][j + 1]  # same row neighbours
    number_alive += grid[i + 1][j - 1] + grid[i + 1][j] + grid[i + 1][j + 1]  # bottom row neighbours
    return number_alive


# this function superimposes the two ideas, kind like a prey and predator system
def update(nb, cell, overpop_threshold=3, underpop_threshold=2, reproduction_threshold=3, mutation_rate=2,
           isHyper=True):
    if cell == 1:
        if nb > overpop_threshold:  # overpopulation
            return 2
        elif nb < underpop_threshold:  # underpopulation
            return 0
        elif nb == overpop_threshold or nb == underpop_threshold:
            return 1
    elif cell == 0:
        if nb == reproduction_threshold:  # reproduction
            return 1

    if isHyper:
        overpop_threshold = overpop_threshold * mutation_rate
        underpop_threshold = (underpop_threshold * mutation_rate) + 1
        if cell >= 2:

            if nb > overpop_threshold or nb < underpop_threshold:  # not viabile
                return 0
            elif nb == overpop_threshold or nb == underpop_threshold:
                return cell + (reproduction_threshold * mutation_rate)

    return cell


def init_frame(grid):
    grid_test = np.array(grid, dtype=float)
    fig, ax = plt.subplots()
    img = ax.imshow(grid_test, cmap='gray')
    return img
    # old code -> plt.imshow(grid_test, cmap='gray')
    # old code  -> plt.show(block = True)]]


# where the game happens
def game(old_grid, step=10, overpop_threshold=3, underpop_threshold=2, reproduction_threshold=3, mutation_rate=2,
         index = "", isHyper=True):
    output = open("output.txt", "w")

    fig = plt.figure()
    ims = []
    grid_test = np.array(old_grid, dtype=float)
    img = (plt.pcolor(grid_test, cmap='Greys'),)  # initializes (or at least it would in theory)  my figure
    ims.append(img)

    t = 1  # starting with 1
    grid = [row[:] for row in old_grid]  # deepcopying my grid

    while t <= step:

        for row in range(1, len(grid) - 1):
            for column in range(1, len(grid[row]) - 1):
                grid[row][column] = update(number_neighbours(row, column, old_grid), old_grid[row][column],
                                           overpop_threshold, underpop_threshold, reproduction_threshold, mutation_rate,
                                           isHyper)

        output.write(str(step) + '\n')  # writes what steps i am in
        for i in range(len(grid)):  # writes the actual grid
            str_test = ""
            for j in range(len(grid[i])):
                str_test += str(grid[i][j])

        grid_test = np.array(grid, dtype=float)
        # img= (plt.pcolor(grid_test, cmap='Greys')) #initializes (or at least it would in theory)  my figure
        ims.append((plt.pcolor(grid_test, cmap='Greys'),))

        old_grid = [row[:] for row in grid]  # deep copying my grid
        t += 1
        # helper_animation +=1

    output.close()
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat=True, repeat_delay=1000)
    #plt.show(block=False)
    saving_code =  index + "movie.mp4"
    ani.save(saving_code)
    plt.close()
    return old_grid


def random_entry(n=5, m=5):
    entry = [[np.random.randint(0, 2) for column in range(m)] for line in range(n)]

    return entry


def fitness(grid, weight_normal=2, weight_hyper=2, weight_tumor = 1):
    # First part, counting how many of each cells I have
    hyper = 0
    normal = 0
    tumor = 0
    total = (len(grid) - 1) ** 2
    for row in range(0, len(grid) - 1): #range = 1 for trials 1 and 2 and 3
        for column in range(0, len(grid[row]) - 1): #range = 1 for trials 1 and 2 and 3
            if grid[row][column] < 1:
                normal = normal + 1
            elif grid[row][column] == 1:
                tumor = tumor + 1
            else:
                hyper = hyper + 1
    # I want at leat 90% to be normal cells
    perc_normal = normal / total
    perc_hyper = tumor / (tumor + normal)
    clear_tumor = (1 - tumor)
    return perc_normal * weight_normal + perc_hyper * weight_hyper + weight_normal * clear_tumor

# def selection():
# Only Best: Only select the best individuals

def mutation(c1, limit=10):
    # Always choose one bit and only invert this bit
    # I'll be trying to mutate only the mutation_rates (no pun intended)
    # my reasoning here is that easier to mutate this because it's not as dependent as the other ones
    c2 = Chromosome(c1.overpop, c1.underpop, c1.reproduction, np.random.randint(0, limit))
    return c2


def crossing_over(c1, c2):
    # Choose a point p which divides the chromosome in two parts: i chose the half because i think it makes more sense
    # the left side is taken from parent A, the right side from parent B.
    c = (Chromosome(c1.overpop, c1.underpop, c2.reproduction, c2.mutation_rate), Chromosome(c2.overpop, c2.underpop, c1.reproduction, c1.mutation_rate))
    i=np.random.randint(0, 2)  # adding some randomness
    return c[i]


def evolution(number_steps=50, popsize=20, gridsize=100, lifespan=50):
    isEvolving = True
    nb = 0
    grid_testing = random_entry(gridsize, gridsize)

    pop = list()  # list of chromosomes
    for i in range(0, popsize):
        pop.insert(i, Chromosome(np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10),
                            np.random.randint(0, 10)))

    while isEvolving and nb < number_steps:
        grid_testing = random_entry(gridsize, gridsize)
        simulation = list()
        fitness_score = list()

        # fitness and simulation
        for i in range(0, len(pop)):
            simulation.insert(i, game(grid_testing, lifespan, pop[i].overpop, pop[i].underpop, pop[i].reproduction,
                                 pop[i].mutation_rate, str(nb) + "_"+ str(i), isHyper=True))
            fitness_score.insert(i, fitness(simulation[i]))

        # parent selection
        #select  best - i need to search the best
        tuple_id = sorted([(x, i) for (i, x) in enumerate(fitness_score)], reverse=True)[:5]  # fitness score coupled with index (index stays the same across simulation)
        parents = list()
        j = 0
        for i in tuple_id:
            parents.insert(j, pop[i[1]])  # the id between pop simulation and fitness is conserved so i can do this
            j = 1 + j

        # mating (crossover and mutation)
        #crossing over
        offspring = list()
        for i in range(0, 5):
            offspring.insert(i, crossing_over(parents[np.random.randint(0, 4)], parents[np.random.randint(0, 4)]))

        #mutation
        mutated = list()
        for i in range(0, 5):
            mutated.insert(i, mutation(parents[np.random.randint(0, 4)]))

        drift = list()  # list of chromosomes
        for i in range(0, 5):
            drift.insert(i, Chromosome(np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10),
                                     np.random.randint(0, 10)))

        # new generation
        pop = offspring + parents + mutated + drift

        nb = nb + 1

    tuple_id = sorted([(x, i) for (i, x) in enumerate(fitness_score)], reverse=True)[0]
    print(tuple_id)
    print("overpop: ", pop[tuple_id[1]].overpop)
    print("underpop: ", pop[tuple_id[1]].underpop)
    print("reproduction: ", pop[tuple_id[1]].reproduction)
    print("mutation rate: ", pop[tuple_id[1]].mutation_rate)

def test_game():
    input_steps = 50
    grid = random_entry(int(100), int(100))

    pop = list()  # list of chromosomes
    for i in range(0, 5):
        pop.insert(i, Chromosome(np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)))

    simulation = list()
    fitness_score = list()

    for i in range(0, len(pop)):
            simulation.insert(i, game(grid, 20, pop[i].overpop, pop[i].underpop, pop[i].reproduction, pop[i].mutation_rate, str(i)))
            fitness_score.insert(i, fitness(simulation[i]) )

    print(fitness_score)

def dna_testing():
    print("testing class")
    t = Chromosome(10, 2, 7, 5)
    print(t.getoverpop())
    print(t.getunderpop())
    print(t.getreproduction())
    print(t.getmutationrate())

    print("test mutation class")
    t = mutation(t)
    print(t.overpop)
    print(t.underpop)
    print(t.reproduction)
    print(t.mutation_rate)

    print("testing dna crossing_over")
    c  = Chromosome(5, 8, 7, 1)
    t = mutation(t)
    n = crossing_over(c, t)
    print(n.overpop)
    print(n.underpop)
    print(n.reproduction)
    print(n.mutation_rate)


#dna_testing()
#test_game()
evolution()
