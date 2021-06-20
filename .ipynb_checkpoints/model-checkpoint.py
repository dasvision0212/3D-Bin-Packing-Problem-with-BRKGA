import math
import copy
import random
import numpy as np
import pandas as pd
import concurrent.futures

INFEASIBLE = 100000

def generateInstances(N = 20, m = 10, cls = 1, heterogeneous=False):
    # N in 20, 50, 80, 100, 120
    # m in 0, 20, 25, 30, 35, 40   
    def ur(lb, ub):
        # u.r. is an abbreviation of "uniformly random". [Martello (1995)]
        value = random.uniform(lb, ub)
        return int(value) if value >= 1 else 1
    
    p = []; q = []; r = []
    for i in range(N):
        # decide instance class
        if random.random() <= 0.6:
            case = cls
        else:
            case = random.choice([i for i in range(1,6) if i != cls])

        L=100; W=100; H=100
        if case == 1:
            p.append(ur(1, 1/2*L))
            q.append(ur(2/3*W, W))
            r.append(ur(2/3*H, H))
        elif case == 2:
            p.append(ur(2/3*L, L))
            q.append(ur(1, 1/2*W))
            r.append(ur(2/3*H, H))
        elif case == 3:
            p.append(ur(2/3*L, L))
            q.append(ur(2/3*W, W))
            r.append(ur(1, 1/2*H))
        elif case == 4:
            p.append(ur(1/2*L, L))
            q.append(ur(1/2*W, W))
            r.append(ur(1/2*H, H))
        elif case == 5:
            p.append(ur(1, 1/2*L))
            q.append(ur(1, 1/2*W))
            r.append(ur(1, 1/2*H))
    
    if heterogeneous:
        L = [ur(50, 200) for j in range(m)]
        W = [ur(50, 200) for j in range(m)]
        H = [ur(50, 200) for j in range(m)]
    else:
        L = [100]*m
        W = [100]*m
        H = [100]*m
    return range(N), range(m), p, q, r, L, W, H

def generateInputs(N, m):
    N, M, p,q,r, L,W,H =generateInstances(N, m)
    inputs = {'v':list(zip(p, q, r)), 'V':list(zip(L, W, H))}
    return inputs


class Bin():
    def __init__(self, V, verbose=False):
        self.dimensions = V
        self.EMSs = [[np.array((0,0,0)), np.array(V)]]
        self.load_items = []
        
        if verbose:
            print('Init EMSs:',self.EMSs)
    
    def __getitem__(self, index):
        return self.EMSs[index]
    
    def __len__(self):
        return len(self.EMSs)
    
    def update(self, box, selected_EMS, min_vol = 1, min_dim = 1, verbose=False):

        # 1. place box in a EMS
        boxToPlace = np.array(box)
        selected_min = np.array(selected_EMS[0])
        ems = [selected_min, selected_min + boxToPlace]
        self.load_items.append(ems)
        
        if verbose:
            print('------------\n*Place Box*:\nEMS:', list(map(tuple, ems)))
        
        # 2. Generate new EMSs resulting from the intersection of the box
        for EMS in self.EMSs.copy():
            if self.overlapped(ems, EMS):
                
                # eliminate overlapped EMS
                self.eliminate(EMS)
                
                if verbose:
                    print('\n*Elimination*:\nRemove overlapped EMS:',list(map(tuple, EMS)),'\nEMSs left:', list(map( lambda x : list(map(tuple,x)), self.EMSs)))
                
                # six new EMSs in 3 dimensions
                x1, y1, z1 = EMS[0]; x2, y2, z2 = EMS[1]
                x3, y3, z3 = ems[0]; x4, y4, z4 = ems[1]
                new_EMSs = [
                    [np.array((x1, y1, z1)), np.array((x3, y2, z2))],
                    [np.array((x4, y1, z1)), np.array((x2, y2, z2))],
                    [np.array((x1, y1, z1)), np.array((x2, y3, z2))],
                    [np.array((x1, y4, z1)), np.array((x2, y2, z2))],
                    [np.array((x1, y1, z1)), np.array((x2, y2, z3))],
                    [np.array((x1, y1, z4)), np.array((x2, y2, z2))]
                ]
                

                for new_EMS in new_EMSs:
                    new_box = new_EMS[1] - new_EMS[0]
                    isValid = True
                    
                    if verbose:
                        print('\n*New*\nEMS:', list(map(tuple, new_EMS)))

                    # 3. Eliminate new EMSs which are totally inscribed by other EMSs
                    for other_EMS in self.EMSs:
                        if self.inscribed(new_EMS, other_EMS):
                            isValid = False
                            if verbose:
                                print('-> Totally inscribed by:', list(map(tuple, other_EMS)))
                            
                    # 4. Do not add new EMS smaller than the volume of remaining boxes
                    if np.min(new_box) < min_dim:
                        isValid = False
                        if verbose:
                            print('-> Dimension too small.')
                        
                    # 5. Do not add new EMS having smaller dimension of the smallest dimension of remaining boxes
                    if np.product(new_box) < min_vol:
                        isValid = False
                        if verbose:
                            print('-> Volumne too small.')

                    if isValid:
                        self.EMSs.append(new_EMS)
                        if verbose:
                            print('-> Success\nAdd new EMS:', list(map(tuple, new_EMS)))

        if verbose:
            print('\nEnd:')
            print('EMSs:', list(map( lambda x : list(map(tuple,x)), self.EMSs)))
    
    def overlapped(self, ems, EMS):
        if np.all(ems[1] > EMS[0]) and np.all(ems[0] < EMS[1]):
            return True
        return False
    
    def inscribed(self, ems, EMS):
        if np.all(EMS[0] <= ems[0]) and np.all(ems[1] <= EMS[1]):
            return True
        return False
    
    def eliminate(self, ems):
        # numpy array can't compare directly
        ems = list(map(tuple, ems))    
        for index, EMS in enumerate(self.EMSs):
            if ems == list(map(tuple, EMS)):
                self.EMSs.pop(index)
                return
    
    def get_EMSs(self):
        return  list(map( lambda x : list(map(tuple,x)), self.EMSs))
    
    def load(self):
        return np.sum([ np.product(item[1] - item[0]) for item in self.load_items]) / np.product(self.dimensions)
    
class PlacementProcedure():
    def __init__(self, inputs, solution, verbose=False):
        self.Bins = [Bin(V) for V in inputs['V']]
        self.boxes = inputs['v']
        self.BPS = np.argsort(solution[:len(self.boxes)])
        self.VBO = solution[len(self.boxes):]
        self.num_opend_bins = 1
        
        self.verbose = verbose
        if self.verbose:
            print('------------------------------------------------------------------')
            print('|   Placement Procedure')
            print('|    -> Boxes:', self.boxes)
            print('|    -> Box Packing Sequence:', self.BPS)
            print('|    -> Vector of Box Orientations:', self.VBO)
            print('-------------------------------------------------------------------')
        
        self.infisible = False
        self.placement()
        
    
    def placement(self):
        items_sorted = [self.boxes[i] for i in self.BPS]

        # Box Selection
        for i, box in enumerate(items_sorted):
            if self.verbose:
                print('Select Box:', box)
                
            # Bin and EMS selection
            selected_bin = None
            selected_EMS = None
            for k in range(self.num_opend_bins):
                # select EMS using DFTRC-2
                EMS = self.DFTRC_2(box, k)

                # update selection if "packable"
                if EMS != None:
                    selected_bin = k
                    selected_EMS = EMS
                    break
            
            # Open new empty bin
            if selected_bin == None:
                self.num_opend_bins += 1
                selected_bin = self.num_opend_bins - 1
                if self.num_opend_bins > len(self.Bins):
                    self.infisible = True
                    
                    if self.verbose:
                        print('No more bin to open. [Infeasible]')
                    return
                    
                selected_EMS = self.Bins[selected_bin].EMSs[0] # origin of the new bin
                if self.verbose:
                    print('No available bin... open bin', selected_bin)
            
            if self.verbose:
                print('Select EMS:', list(map(tuple, selected_EMS)))
                
            # Box orientation selection
            BO = self.selecte_box_orientaion(self.VBO[i], box, selected_EMS)
                
            # elimination rule for different process
            min_vol, min_dim = self.elimination_rule(items_sorted[i+1:])
            
            # pack the box to the bin & update state information
            self.Bins[selected_bin].update(self.orient(box, BO), selected_EMS, min_vol, min_dim)
            
            if self.verbose:
                print('Add box to Bin',selected_bin)
                print(' -> EMSs:',self.Bins[selected_bin].get_EMSs())
                print('------------------------------------------------------------')
        if self.verbose:
            print('|')
            print('|     Number of used bins:',self.num_opend_bins)
            print('|')
            print('------------------------------------------------------------')
    
    # Distance to the Front-Top-Right Corner
    def DFTRC_2(self, box, k):
        maxDist = -1
        selectedEMS = None

        for EMS in self.Bins[k].EMSs:
            D, W, H = self.Bins[k].dimensions
            for direction in [1,2,3,4,5,6]:
                d, w, h = self.orient(box, direction)
                if self.fitin((d, w, h), EMS):
                    x, y, z = EMS[0]
                    distance = pow(D-x-d, 2) + pow(W-y-w, 2) + pow(H-z-h, 2)

                    if distance > maxDist:
                        maxDist = distance
                        selectedEMS = EMS
        return selectedEMS

    def orient(self, box, BO=1):
        d, w, h = box
        if   BO == 1: return (d, w, h)
        elif BO == 2: return (d, h, w)
        elif BO == 3: return (w, d, h)
        elif BO == 4: return (w, h, d)
        elif BO == 5: return (h, d, w)
        elif BO == 6: return (h, w, d)
        
    def selecte_box_orientaion(self, VBO, box, EMS):
        # feasible direction
        BOs = []
        for direction in [1,2,3,4,5,6]:
            if self.fitin(self.orient(box, direction), EMS):
                BOs.append(direction)
        
        # choose direction based on VBO vector
        selectedBO = BOs[math.ceil(VBO*len(BOs))-1]
        
        if self.verbose:
            print('Select VBO:', selectedBO,'  (BOs',BOs, ', vector', VBO,')')
        return selectedBO
    
    def fitin(self, box, EMS):
        # all dimension fit
        for d in range(3):
            if box[d] > EMS[1][d] - EMS[0][d]:
                return False
        return True
    
    def elimination_rule(self, remaining_boxes):
        if len(remaining_boxes) == 0:
            return 0, 0
        
        min_vol = 999999999
        min_dim = 9999
        for box in remaining_boxes:
            # minimum dimension
            dim = np.min(box)
            if dim < min_dim:
                min_dim = dim
                
            # minimum volume
            vol = np.product(box)
            if vol < min_vol:
                min_vol = vol
        return min_vol, min_dim
    
    def evaluate(self):
        if self.infisible:
            return INFEASIBLE
        
        leastLoad = 1
        for k in range(self.num_opend_bins):
            load = self.Bins[k].load()
            if load < leastLoad:
                leastLoad = load
                
        return self.num_opend_bins + leastLoad
    


class BRKGA():
    def __init__(self, inputs, num_generations = 200, num_individuals=120, num_elites = 12, num_mutants = 18, eliteCProb = 0.7, multiProcess = False):
        # Setting
        self.multiProcess = multiProcess
        # Input
        self.inputs =  copy.deepcopy(inputs)
        
        # increase container number for improvement
        y_num = len(self.inputs['V'])
        for i in range(y_num):
            self.inputs['V'].append(self.inputs['V'][0])
        
        self.N = len(inputs['v'])
        
        # Configuration
        self.num_generations = num_generations
        self.num_individuals = int(num_individuals)
        self.num_gene = 2*self.N
        
        self.num_elites = int(num_elites)
        self.num_mutants = int(num_mutants)
        self.eliteCProb = eliteCProb
        
        # Result
        self.used_bins = -1
        self.solution = None
        self.best_fitness = -1
        self.history = {
            'max' :[],
            'mean': [],
            'min': []
        }
        
    def decoder(self, solution):
        placement = PlacementProcedure(self.inputs, solution)
        return placement.evaluate()
    
    def cal_fitness(self, population):
        fitness_list = list()
        
        return fitness_list
    
    def cal_fitness(self, population):
        fitness_list = list()

        for solution in population:
            decoder = PlacementProcedure(self.inputs, solution)
            fitness_list.append(decoder.evaluate())
        return fitness_list

    def partition(self, population, fitness_list):
        sorted_indexs = np.argsort(fitness_list)
        return population[sorted_indexs[:self.num_elites]], population[sorted_indexs[self.num_elites:]]
    
    def crossover(self, elite, non_elite):
        # chance to choose the gene from elite and non_elite for each gene
        return [elite[gene] if np.random.uniform(low=0.0, high=1.0) < self.eliteCProb else non_elite[gene] for gene in range(self.num_gene)]

    def mating(self, elites, non_elites):
        # biased selection of mating parents: 1 elite & 1 non_elite
        num_offspring = self.num_individuals - self.num_elites - self.num_mutants
        return [self.crossover(random.choice(elites), random.choice(non_elites)) for i in range(num_offspring)]
        
    def fit(self, patient = 4, verbose = False):
        # Initial population & fitness
        population = np.random.uniform(low=0.0, high=1.0, size=(self.num_individuals, self.num_gene))
        fitness_list = self.cal_fitness(population)
        
        

        if verbose:
            print('\nInitial Population:')
            print('  ->  shape:',population.shape)
            print('  ->  Best Fitness:',max(fitness_list))
            
        # best    
        best_fitness = np.min(fitness_list)
        best_solution = population[np.argmin(fitness_list)]
        self.history['min'].append(np.min(fitness_list))
        self.history['mean'].append(np.mean(fitness_list))
        self.history['max'].append(np.max(fitness_list))
        
        
        # Repeat generations
        best_iter = 0
        for g in range(self.num_generations):

            # early stopping
            if g - best_iter > patient:
                self.used_bins = math.floor(best_fitness)
                self.best_fitness = best_fitness
                self.solution = best_solution
                if verbose:
                    print('Early stop at iter', g, '(timeout)')
                return 'feasible'

            # infeasible
            if np.min(fitness_list) == INFEASIBLE:
                if verbose:
                    print('INFEASIBLE', np.min(fitness_list))
                return 'infeasible'
            
            # Select elite group
            elites, non_elites = self.partition(population, fitness_list)
    
            # Generate mutants
            mutants = np.random.uniform(low=0.0, high=1.0, size=(self.num_mutants, self.num_gene))
            
            # Biased Mating & Crossover
            offsprings = self.mating(elites, non_elites)

            # New Population & fitness
            population = np.concatenate((elites,mutants,offsprings), axis=0)
            
            fitness_list = self.cal_fitness(population)
            
            # Update Best Fitness
            for fitness in fitness_list:
                if fitness < best_fitness:
                    best_iter = g
                    best_fitness = fitness
                    best_solution = population[np.argmin(fitness_list)]
            
            self.history['min'].append(np.min(fitness_list))
            self.history['mean'].append(np.mean(fitness_list))
            self.history['max'].append(np.max(fitness_list))
            
            if verbose:
                print("Generation :", g, ' \t(Best Fitness:', best_fitness,')')
            
        self.used_bins = math.floor(best_fitness)
        self.best_fitness = best_fitness
        self.solution = best_solution
        return 'feasible'

# import pandas as pd
# import time

# if __name__ == '__main__':
#     e = pd.DataFrame(columns = ['N', 'size', 'i','time_multi','time_serial'])
#     for n in [50, 70, 100]:
#         print(n)
#         for i in range(5):

#             time_multi = time.time()
#             inputs = generateInputs(n,int(n/2))
#             model = BRKGA(inputs, num_generations = 10, num_individuals=n*10, num_elites =n*1, num_mutants = n*1.5, eliteCProb = 0.7)
#             model.fit(multiProcess=True, verbose = False)
#             time_multi = time.time() - time_multi

#             time_serial = time.time()
#             inputs = generateInputs(n,int(n/2))
#             model = BRKGA(inputs, num_generations = 10, num_individuals=n*10, num_elites =n*1, num_mutants = n*1.5, eliteCProb = 0.7)
#             model.fit(multiProcess=False, verbose = False)
#             time_serial = time.time() - time_serial

#             e = e.append({
#                 'N': n,
#                 'size': n*10,
#                 'i': i,
#                 'time_multi': time_multi,
#                 'time_serial': time_serial
#             }, ignore_index=True)
#     e.to_csv('experiments.csv', index = False)
#     print('FINISh')