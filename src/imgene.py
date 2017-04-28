#! /usr/bin/env python

'''
Optimizes an image (starting as random pixels) to become the same image as another

@author: hdamron1594
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2

from os.path import join as pathjoin, abspath, dirname
from functools import reduce
import sys
import random
import warnings


warnings.filterwarnings("ignore")
PROJECT_ROOT =  pathjoin(dirname(abspath(sys.argv[0])), "..")

def load_img(filename, greyscale=True):
    im = cv2.imread(filename, 0 if greyscale else -1) #greyscale else unchanged
    if im is None:
        raise UserWarning("File not found %s" % filename)
    else:
        return im

def random_img(size):
    return np.random.randint(low=0, high=255, size=size)
    
def compare_img(im1, im2):
    total = 0
    for this, that in zip(np.nditer(im1), np.nditer(im2)):
        total += abs(this - that)
    return total

def first_gen(model_im, num=20):
    '''
    Creates generation of random images in shape of model_im
    :param model_im: model greyscale image to create
    :param num: number of items in generation
    :return: returns list of tuples (fitness, image)
    '''
    gen = []
    for i in range(num):
        rand_im = random_img(model_im.shape)
        fitness = compare_img(rand_im, model_im)
        gen.append((fitness, rand_im))
    return gen

def mate(mom, dad):
    '''
    Mating function between two images (must have same shape)
    For each index in the two images, one of the two is picked randomly
    :param mom: first image parent
    :param dad: second image parent
    :return: returns a new image of same shape with attributes from both parents
    '''
    assert mom.shape == dad.shape, "Images must have same shape"
    shape = mom.shape
    baby = np.zeros(shape)
    for index in np.ndindex(*shape):
        #tuple index ued to index mom and dad
        baby[index] = random.choice((mom[index], dad[index])) #replace baby attribute with one from mom or dad
    return baby

def mutate_element(im, rate, std_dev=20):
    '''
    Elementwise mutation of imgene
    :param im: actually a single element but a 2D image because of vectorization
    :param rate: probability of mutation in range [0, 1)
    :param changes: possible changes in state
    :return: returns image of same shape as im
    '''
    if random.random() <= rate:
        #this one gets changed
        return im + random.gauss(0, std_dev) #TODO possibly use a different random for speed
    else:
        return im #no change

mutate = np.vectorize(mutate_element, excluded=['rate', 'changes'])

def mate_and_mutate(mom, dad, mutation_rate, std_dev=20):
    '''
    Applies mating and mutating in one function since they are commonly used together
    :param mom: first image parent
    :param dad: second image parent
    :param im: actually a single element but a 2D image because of vectorization
    :param rate: probability of mutation in range [0, 1)
    :param changes: possible changes in state
    :return: returns offspring after mutation
    '''
    return mutate(mate(mom, dad), mutation_rate, std_dev)

def evolve(model_im, goal_fitness_per_pixel, pop_size=20, mutation_rate=0.01, std_dev=20, show=False, elite=True):
    '''
    Runs successive iterations of genetic algorithm until it finds an image with fitness less than the cap
    :param model_im: image to emulate
    :param goalfitness: maximum fitness level to allow in return
    :param pop_size: number of items in each population
    :param mutation_rate: decimal number [0, 1) for mutation rate
    :param std_dev: Standard deviation for mutation
    :return: returns ideal image
    '''
    pop = first_gen(model_im, num=pop_size)
    pixels = reduce(lambda a,b: a*b, model_im.shape)
    goal_fitness = goal_fitness_per_pixel * pixels
    
    if show:
        print("Info: Drawing Initial Screen")
        display = plt.imshow(pop[0][1], cmap="gray", vmin=0, vmax=255)
        plt.ion()
        plt.show()
    
    best = min(pop)
    generation = 1
    while best[0] > goal_fitness:
        fitnesses = [element[0] for element in pop] #get fitness levels of each
        
        num = pop_size if not elite else pop_size-1
        new_pop_items = [mate_and_mutate(pop[i][1], pop[j][1], mutation_rate, std_dev) for i,j in pairings(fitnesses, num=num)]
        new_pop_fitnesses = [compare_img(im, model_im) for im in new_pop_items]
        
        pop = list(zip(new_pop_fitnesses, new_pop_items))
        best = min(pop)
        
        if elite:
            pop.insert(0, best)
        
        print("Info: Gen %s Best Fitness = %s" % (generation, best[0] / pixels))
        generation += 1
        
        if show and len(plt.get_fignums()) > 0:
            #If window is still open, it prints
            try:
                display.set_data(best[1])
                plt.draw()
                plt.pause(0.01)
            except Exception as e:
                #If anything fails, we just rage quit and kill everything
                show = False
                plt.close('all')
                print("Warning: Window failure with exception %s" % e)

    return best[1]
    
def pairings(fitnesses, num):
    '''
    Generates index pairings of generations
    :param fitnesses: list of fitness levels to use as weights for randomness
    :return: returns list of integer 2-tuples with indices of pairs to mate
    '''
    assert len(fitnesses) > 1, "Can't make pair of 1 item"
    pairs = []
    indices = list(range(len(fitnesses))) #get list of possible indices
    worst = max(fitnesses)
    inversed_fitnesses = worst - fitnesses
    weights = inversed_fitnesses / sum(inversed_fitnesses)
    
    while len(pairs) < num:
        pairs.append(np.random.choice(indices, size=2, replace=True, p=weights)) #2-tuple with 2 randomly chosen indices
    return pairs

def test_main():
    im = load_img(pathjoin(PROJECT_ROOT, "images", "monalisa.jpg"))
    plt.imshow(im, cmap="gray")
    #plt.show()
    
    recreated = evolve(im, goal_fitness_per_pixel=5, pop_size=30, mutation_rate=0.005, std_dev=20, show=True, elite=True)
    plt.ioff()
    plt.imshow(recreated, cmap="gray", vmin=0, vmax=255)
    plt.show()

if __name__ == "__main__":
    test_main()

