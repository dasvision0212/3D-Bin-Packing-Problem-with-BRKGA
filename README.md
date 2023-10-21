# Tutorial of BRKGA for 3D-Bin Packing Problem

<details open="open">
  <summary><b>Table of Contents</b></summary>
  <ol>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li>
      <a href="#problem-description">Problem Description</a>
    </li>
    <li>
      <a href="#methodology">Methodology</a>
      <ul>
        <li><a href="#biased-random-key-enetic-lgorithmn">Biased Random-Key Genetic Algorithmn</a></li>
        <li><a href="#placement-strategy">Placement Strategy</a></li>
      </ul>
    </li>
    <li><a href="#visualization">Visualization</a></li>
    <li><a href="#conclusion">Conclusion</a></li>
  </ol>
</details>

# __Introduction__

This repository provides a tutorial on the implementation of the __Biased Random Key Genetic Algorithmn for 3D Bin Packing Problem__. It is inspired by the heuristic and design proposed in the paper _"A biased random key genetic algorithm for 2D and 3D bin packing problems"_  by [Gonçalves & Resende (2013)]("https://www.sciencedirect.com/science/article/abs/pii/S0925527313001837?via%3Dihub"). I created this tutorial as a practical application for the course "Operations Research Applications and Implementation (IM5059 2020-Spring)" taught by Professor Chia-Yen Lee at National Taiwan University. 

(Update in 10.2023)

## __Outline__

We'll begin by outlining the problem statement for the __3D Bin Packing Problem (3D-BPP)__, an extension of a classic NP-hard problem of (1D) bin packing problem. Then we'll delve into the Biased Random-Key Genetic Algorithm (BRKGA) and the Placement Strategy as put forth in the aforementioned paper, supplemented with code snippets and illustrative examples. To conclude, we'll showcase visualizations of sample solutions.

__ROLL TO THE END OF EACH CHAPTER IF YOU WANT TO SKIP THE EXPLAINATION__

## __Prerequisites__

- Mostly implemented with NumPy that can commonly be found in most Python distributions. No other external libraries is required.

- It is recommended that readers possess a foundational knowledge of genetic algorithms. This includes familiarity with concepts like chromosome representation, encoding and decoding processes, and evolutionary cycle. I recommend reading this [article](https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_quick_guide.htm) from Tutorialspoint.

# __Problem Description__

## __Three Dimensional Bin-Packing Problem__

<p style="text-align:center">
  <img src="./resource/images/3dbpp.jpg" />
  <center>Fig. example of bin packing problem, from <a href="https://www.sciencedirect.com/science/article/abs/pii/S0925527313001837?via%3Dihub">[Gonçalves & Resende (2013)]</a></center>
</p>

In 1D bin-packing problem, __items with varying sizes__ are packed into multiple containers (called bins) usually of equal capacity. The primary objective is to __minimize the number of bins used__, which differentiates it from the multiple knapsack problem where the number of bins is fixed.

In three dimensional bin-packing problem (3D-BPP), we introduces additional spacial constaints, taking into account the __length, width, height__ of an item. Items with different volumes must be orthogonally packed into fixed size bins __without overlapping__.

The problem is strongly __NP-hard__, as it generalizes from the 2D-bpp and 1D-bpp, both of which are also NP-hard. This implies that there are no known polynomial-time solutions for this problem. Even though a mixed-integer programming formulation exists that can guarantee an optimal solution, a faster heuristic is often preferred due to the computational intensity of such methods.

With this in mind, the __genetic algorithm__, a prevalent heuristic for combinatorial problems, can be employed to address the problem in a reasonable timeframe. However, it's important to note that this approach does __not guarantee an optimal solution__.

## __Problem Definition__

- A set of _n_ rectangular items (boxes). Each item _i_ has dimensions $(d_i, w_i, h_i)$, representing width, height, and depth.
- A set of identical bins (containers) with demensions $(W_j, H_j, D_j)$ represeting dimensions of $j_{th}$ bin.

For this implementation, we use two array of tuples to denote dimensions of boxes and containers respectively. Here we relax the assumption of equal size container for generalization.

```python
inputs: {
 # boxes with different shape
 'v': [(188, 28, 58), (61, 9, 79), (188, 28, 58), ..., (145, 80, 96)],
 # containers with fixed shape
 'V': [(610, 244, 259), .., (610, 244, 259)]
}
```

# __Methodology__

The proposed algorithm consists of two main components: **Biased Random-Key Genetic Algorithm (BRKGA)**, and **placement procedure** heuristic. In essence, the algorithm encodes each solution (or packing scenario) using a sequence/vectors. This encoding allows **genetic algorithm** to search optimal solutions via selection.

In the following section, we will delve into the specifics of the BRKGA settings and the underlying concept of the placement procedure.

<p style="text-align:center">
  <img src="./resource/images/architecture.png" width="400" />
  <center>algorithmn architecture</center>
</p>

## __Biased Random-Key Genetic Algorithmn__

The Biased Random-Key Genetic Algorithm (BRKGA) is fundamentally a variation of the genetic algorithm (GA) that incorporates the features of random-key representation and biased selection from the population pool.

> **Note:** It's important to highlight that in most 3D-BPP papers, rather than detailing every coordinate of placed items, they focus on determining the __optimal packing sequence__ (the specific order in which items are packed). This often follows a rule-based packing procedure, such as the Deepest Bottom-Left-Fill packing method.
>
> By adopting these approaches, the solution's representation becomes significantly more streamlined, and, based on empirical evidence, high-quality solutions are often achieved. Furthermore, this packing sequence can invariably be translated into a viable packing scenario, eliminating concerns about coordinate overlaps. Consequently, there's no need for chromosome-repair in a GA iteration. Of course, these rule-based packing prcedure do not gaurantee an optimal way of packinig.

### __Random-Key Representation__

The random key is a way to encode a chromosome (solution) as a vector of real numbers within $[0, 1]$. Let assume there are _n_ items to be placed. The length of a random key will be _2n_. We can initialize an solution as follow:

```python
# random key representation
solution = np.random.uniform(low=0.0, high=1.0, size= 2*n)

# e.g., there are 4 items to be packed, the frist n (=4) values of a random key may look like [0.1, 0.5, 0.6, 0.2] 
```

In each random-key, the first _n_ genes encode the order of the _n_ items to be packed, which is called **Box Packing Sequence (BPS)**. We decode it by sorting it in ascending order of the corresponding gene values (Fig. 3). In NumPy,  We can use the _argsort_ method to obtain the indices of the sorted array. 

```python
# sort by ascending order
box_packing_sequence = np.argsort(solution[:n])

# e.g., after decoded, the BPS for the above example will be [0, 2, 3, 1].
# The first box will be packed first, and the third box the last.
```



<p style="text-align:center">
  <img src="./resource/images/BPS.png" />
  <center><b>Fig. 3.</b> Decoding of the box packing sequence</center>
</p>
<p style="text-align:center">
  <img src="./resource/images/box_orientation.png" />
  <center><b>Fig. 4.</b> Box orientation</center>
</p>

The last _n_ genes of a random key encode the orientation of items - **Vector of Box Orientations (VBO)**. In three dimension setting, there are in total __six__ orientations (Fig. 4) to place a item. In some scenarios, items are limited by the vertical orientation (cannot be placed upside down). We can create a circular mapping for possible orientation using Python list. The decoding of each gene (called _BO_) in _VBO_ is defined as:

<p align="center">
  <b>selected orientation = BOs⌈BO×nBOs⌉</b><br>
</p>

, where _BO_ is the last _n_ genes, _BOs_ denotes all possible orientations allowed for that box, and _nBOs_ is the number of _BOs_.

```python
# value of BO
BO = 0.82

# posiible orientations
BOs = [1,2,3,4,5,6]  # remove 2, 5 to remove vertical orientations

# selected orientation
orientation = BOs[math.ceil(BO*len(BOs))] # orientation = 5
```
This mapping will ensure a _BO_ value corresponds to a possible orientation.

In summary, _BRKGA_ use a vector of real numbers to encode a solution. The first _n_ genes represent __box packing sequence (BPS)__, the order to pack the items. The rest _n_ genes represent __Vector of Box Orientations (VBO)__, the oreientation of those corresponding items. With a packing procedure that will be explained later, this vector can always be converted to a real packing scenario.

### __Biased Selection__

In each generation, the population is partitioned into two group, __Elite__ and __Non-Elite__, based on the fitness value. This biased selection will greatly influence operations in GA such as crossover and survivor selection that will be explained latter.  For now, let's define a function to partition the population into elite and non-elite group.

```python
def partition(population, fitness_list, num_elites):
    # sorted indices based on fitness value
    sorted_indexs = np.argsort(fitness_list)
    # return elite & non-elite group
    return population[sorted_indexs[:num_elites]], population[sorted_indexs[num_elites:]]
```

The _fitness_list_ is the list of fitness value for all solutions after evaluation. After sorting, we can simply use NumPy indexing to partition two group.

### __Crossover__

In _BRKGA_, the __parameterized uniform crossover__ is used to implement the crossover operation. For each mating, we always chose one parent from the _Elite_ group and the other from  the _Non-Elite_. Each offspring inherited _i-th_ gene from either elite or non-elite based on a prespecified probality denoted as _eliteCProb_. This probability control the extend of favoring the inheritance from the elite parent.

```python
def crossover(elite, non_elite):
    # initialize chromosome
    offspring = [0]*(2*n)

    # choose each gene from elite and non_elite
    for i in range(2*n):
      # inherit from elite with probability of eliteCProb
      if np.random.uniform(low=0.0, high=1.0) < eliteCProb:
          offspring][i] = elite[i]
      else: 
          offspring][i] = non_elite[i]
    return offspring

def mating(self, elites, non_elites):
    offspring_list = []
    num_offspring = num_individuals - num_elites - num_mutants
    for i in range(num_offspring):
        # biased selection for parents: 1 elite & 1 non_elite
        offspring = crossover(random.choice(elites), random.choice(non_elites))
        offspring_list.append(offspring)
    return offspring_list
```

The number of offsprings (from crossover) is the complement ofthe number of individuals (_num_individuals_) and the number of mutants (_num_mutants_).

### __Mutants__

Instead of performing common mutation operation (e.g. swap mutation), the paper create entire new individuals to increase random noise into the population (complex problem require more noise, in a sense). Given the number of mutants, denoted as _num_mutants_, we can define a function to create new individuals like how we initialize the population.

```python
def mutation(num_mutants):
    # return mutants
    return np.random.uniform(low=0.0, high=1.0, size=(num_mutants, 2*n))
```

### __Evolutionary Process__

In each generation, all elite individuals survive and move to the next population without any modification; mutants and offsprings are also added to the next population. Then we will update the  fitness value in each generation:

```python
def evolutionary_process(n, num_generations, num_individuals, num_elites, num_mutants):
    
    # initialization
    ## initial poulation
    population = np.random.uniform(low=0.0, high=1.0, size=(num_individuals, 2*n))
    
    ## calculate fitness function
    fitness_list = cal_fitness(population)
    
    ## minimum fitness value
    best_fitness = min(fitness_list)

    for g in range(num_generations):
        
        # seperate elite group & non-elite group
        elites, non_elites = partition(population, fitness_list, num_elites)
        
        # biased mating & crossover
        offsprings = mating(elites, non_elites)

        # generate mutants
        mutants = mutation(num_mutants)

        # next population
        population = np.concatenate((elites, mutants, offsprings), axis=0)
        
        # calculate fitness
        fitness_list = cal_fitness(population)

        # update minimum fitness value
        for fitness in fitness_list:
            if fitness < best_fitness:
                best_fitness = fitness
```

We have now discussed most mechanisms related to GA operations. The remaining piece of the puzzle is understanding how the fitness function is evaluated.

Even with the same packing order (BPS) and orientation (VBO), **different packing strategies can yield varying results** (e.g., you can prioritize putting items in the corder, or finishing a stack first; Like playing Tetris, you have different strategies). In the paper, we adopted _Placement Procedure_, a way to pack the items.

## __Placement Strategy__

In terms of implementation and intricacy, the placement procedure is actually more complex than GA. In previous chapter, we use packing order to represent a packing scenario. In the evaulation phase, we still requires a coordinate system to reflect overlapping and out-of-bound condition for every items placed. The subsequent chapter will explain a placement method proposed in the paper.

First we explain how we model the coordinate system, and determine overlapping condition.

### __Maximal-Spaces Representation__

The maximal-space is a way to represent a rectangular space by its _minimum_ and _maximum coordinats_. This only works if all objects is placed orthogonally, as in this problem. For example, a box with shape of (10, 20, 30) placed in the origin can be encoded as:

```python
# placed at origin (0,0)
MS = [(0,0,0), (10,20,30)]  # [min coordinats, max coordinates] or essentially [min coordinates, min coordinates + shape]
```

We can model the position of each item using this representation.

### __Empty Mximal-Space__

Besides items, We need to model the remaining space. The heuristic — __difference process (DP)__ — developed by [Lai and Chan (1997)](https://www.sciencedirect.com/science/article/abs/pii/S0360835296002057) keeps track available spaces in the container after each box placement. The set of available empty space is called _Empty Maximal-Spaces (EMSs)_. The algorithm goes as follow:
  
If one chose to place box at _EMS_ from existing _EMSs_ ($EMS \in EMSs$). 
  1. Generate 6 new _EMS_ from the intersection of the box with existing EMS and remove the intersected _EMS_ from the set _EMSs_.
  2. Remove new _EMS_ that have infinite thinness (no length in any dimension), or are totally inscribed by other _EMSs_
  3. Remove _EMSs_ which are smaller than existing boxes to be placed.

For each pair of box (demote its space as _ems_) and intersected _EMS_ in _step 2._,  we can compute new  _EMSs_ as follow. Notice that there will be six empty space generated by the intersection in three dimentional space.

```python
# Remember we use Maximal Space representation
# EMS = [
#  (x1, y1, z1),  ## minimum coordinates
#  (x2, y2, z2),  ## minimum coordinates
#]

# minimem & maximum coordinates for intersected EMS
x1, y1, z1 = EMS[0]
x2, y2, z2 = EMS[1]

# minimem & maximum coordinates box to be placed
x3, y3, z3 = ems[0]
x4, y4, z4 = ems[1]

# six new EMSs for 3D space
new_EMSs = [
    [(x1, y1, z1), (x3, y2, z2)],
    [(x4, y1, z1), (x2, y2, z2)],
    [(x1, y1, z1), (x2, y3, z2)],
    [(x1, y4, z1), (x2, y2, z2)],
    [(x1, y1, z1), (x2, y2, z3)],
    [(x1, y1, z4), (x2, y2, z2)]
]
```

In practice, we will place the minimum coordinates of box against the minimum coordiantes of the selected _EMS_ (Fig. 5), or in other word, corner to corner. If we chose to place at abitrary space within the EMS, new EMSs often poorly disjointed and lead to inefficient space utilization. Also, if we chose to place corner to corner, we can omit 3 operations of finding intersections demonstrated as follow.

```python
# minimem & maximum coordinates for intersected EMS
x1, y1, z1 = EMS[0]
x2, y2, z2 = EMS[1]

# minimem & maximum coordinates for space of box
x3, y3, z3 = ems[0]
x4, y4, z4 = ems[1]

# three new EMSs for 3D space if ems[0] = EMS[0]
new_EMSs = [
    [(x4, y1, z1), (x2, y2, z2)],
    [(x1, y4, z1), (x2, y2, z2)],
    [(x1, y1, z4), (x2, y2, z2)]
]
```
<p style="text-align:center">
  <img src="./resource/images/EMS.png" />
  <center><b>Fig. 5.</b> Example of Difference Process (rectangle with bold lines are the new EMSs resulting from the placement of the grey box)</center>
</p>

To check if a _EMS_ overlaps or is totally inscribed by another _EMS_, we can use following functions. (every _EMS_ is converted into _numpy_ array to ultilize element-wise boolean operations)

```python
def overlapped(EMS_1, EMS_2):
    if np.all(EMS_1[1] > EMS_2[0]) and np.all(EMS_1[0] < EMS_2[1]):
        return True
    return False

def inscribed(EMS_1, EMS_2):
    if np.all(EMS_2[0] <= EMS_1[0]) and np.all(EMS_1[1] <= EMS_2[1]):
        return True
    return False
```

Combining all above, the psuedo function for dfference process of a box placement with selected EMS can be written as:

```python
def difference_process(box, selected_EMS, existing_EMSs):

    # 1. compute maximal-space for box with selected EMS
    ems = [selected_EMS[0], selected_EMS[0] + boxToPlace]

    # 2. Generate new EMSs resulting from the intersection of the box 
    for EMS in existing_EMSs:
        if overlapped(ems, EMS):
          
          # eliminate overlapped EMS
          existing_EMSs.remove(EMS)

          # three new EMSs in 3D
          x1, y1, z1 = EMS[0]; x2, y2, z2 = EMS[1]
          x3, y3, z3 = ems[0]; x4, y4, z4 = ems[1]
          new_EMSs = [
              [(x4, y1, z1), (x2, y2, z2)],
              [(x1, y4, z1), (x2, y2, z2)],
              [(x1, y1, z4), (x2, y2, z2)]
          ]

          for new_EMS in new_EMSs:
              isValid = True

              # 3. Eliminate new EMSs which are totally inscribed by other EMSs
              for other_EMS in self.EMSs:
                  if self.inscribed(new_EMS, other_EMS):
                      isValid = False
              
              # 4. Remove _EMSs_ which are smaller than existing boxes to be placed
              ## (1) new EMS smaller than the volume of remaining boxes
              new_box = new_EMS[1] - new_EMS[0]
              if np.min(new_box) < min_dim:
                  isValid = False
              ## (2) new EMS having smaller dimension of the smallest dimension of remaining boxes
              if np.product(new_box) < min_vol:
                  isValid = False

              # add new EMS if valid
              if isValid:
                  existing_EMSs.append(new_EMS)
```

Now we have a way to update the state of container and items in 3D coordinate for each box placement. Next, we introduce a placement heuristic to decide which EMS to select for each box placement.

### __Placement Heuristic__

The _Back-Bottom-Left-Fill Heuristic_ is a rule to pack a sequence of boxes, in which it will always select the empty space with smallest minimum coordinates to fit the box. The heuristic aims to place box in the deepest space for each iteration in hope that all boxes will be placed tight together.

As observed by [Liu and Teng (1999)](https://www.sciencedirect.com/science/article/abs/pii/S0377221797004372), some optimal solutions could not be constructed by this heuritic. To deal with this problem, [Gonçalves & Resende (2013)]("https://www.sciencedirect.com/science/article/abs/pii/S0925527313001837?via%3Dihub") developed an improved version of the placement heuristic rule named _Distance to the Front-Top-Right Corner (DFTRC)_. As the title suggests, the heuristc will always place the box in the empty space such that it maximizes the distance of the box to the maximal coordinates of the container (Fig. 6). You can think of it define "deepest" using maximum distance from the top instead of minimum distance to the corner.

<p style="text-align:center">
  <img src="./resource/images/DFTRC.png" />
  <center><b>Fig. 6.</b> Example of heuristic DFTRC placement rule</center>
</p>

The psuedo function for DFTRC placement rule is:

```python
# D, W, H are the depth, width and height of a container
def DFTRC(box, existing_EMSs):
    maxDist = -1
    selectedEMS = None
    for EMS in existing_EMSs:

        # for different orientation
        for direction in [1,2,3,4,5,6]:
            d, w, h = orient(box, direction)

            # if box fit in the current EMS
            if fitin((d, w, h), EMS):

                # minimum coordinate of ENS
                x, y, z = EMS[0]

                # distance between maximum coordinate of box and container
                distance = pow(D-x-d, 2) + pow(W-y-w, 2) + pow(H-z-h, 2)

                # find maximal distance
                if distance > maxDist:
                    maxDist = distance
                    selected_EMS = EMS
    return selected_EMS
```
where _orient_ is a helper function to orient box given the orientation and _fitin_ to check whether the space can fit in a EMS:

```python
def orient(box, BO):
    d, w, h = box
    # rotate box based on selected orientation BO
    if   BO == 1: return (d, w, h)
    elif BO == 2: return (d, h, w)
    elif BO == 3: return (w, d, h)
    elif BO == 4: return (w, h, d)
    elif BO == 5: return (h, d, w)
    elif BO == 6: return (h, w, d)

def fitin(self, box, EMS):
    # ensure box is totally inscribed by EMS
    for d in range(3):
        if box[d] > EMS[1][d] - EMS[0][d]:
            return False
    return True
```

### __Placement Procedure__

We applied _DFTRC_ placement rule when selecting _EMS_ from existing _EMSs_ for box placement. For each solution, this placement rule is called  _n_ times to place _n_ boxes following the order of _BPS_. If the box cannot fit in the existing _EMSs_, __we will open a new empty container and resume the ongoing placement process__. We can finally finish the __placement procedure__. Let _boxes_ be the set of boxes , _Bins_ be the set of containers, and _num_opend_bins_ be the number of currently opened containers.

```python
def placement_procedure(BPS, VBO):

    # pack box in the order of BPS
    items_sorted = [boxes[i] for i in BPS]

    for i, box in enumerate(items_sorted):
            
        # selection Bin and EMS to place the box
        selected_bin = None
        selected_EMS = None
        for k in range(num_opend_bins):
            
            # select EMS using DFTRC heuristic rule
            EMS = DFTRC(box, Bins[k].existing_EMSs)

            # select successfully
            if EMS != None:
                selected_bin = k
                selected_EMS = EMS
                break
        
        # Open new empty bin if failed
        if selected_bin == None:
            num_opend_bins += 1
            selected_bin = num_opend_bins - 1
            # select the first and only EMS from the new Bin
            selected_EMS = Bins[selected_bin].EMSs[0]

        # Box orientation selection
        BO = selecte_box_orientaion(VBO[i], box, selected_EMS)
            
        # pack the box to the bin & update state information
        # remember it is perform on 
        difference_process(orient(box, BO), selected_EMS, Bins[selected_bin].existing_EMSs)
```

where _selecte_box_orientaion_ is the function to compute and select the orientations for the box:

```python
def selecte_box_orientaion(BO, box, selected_EMS):
    
    # compute possible direction
    BOs = []
    for direction in [1,2,3,4,5,6]:
        if fitin(orient(box, direction), selected_EMS):
            BOs.append(direction)
    
    # select orientation (decoding of BO)
    return BOs[math.ceil(VBO*len(BOs))-1]
```


### Adjusted Fitness Value

One can simply use the number of used bins as the fitness value for each packing scenario. This is totally appropriate and alligns to our objective of minimizing number of used bins. However,  ㄍpaper suggested we can improve the packing "quality" by macking a small adjustment to the fitness value.

Besides number of used bins, we add the percentage ultilization of the least laoded container (Eq. 1) to the fintess value. The rationale for this measure is that given two solutions that have same number of containers, the one having the least load in the least loaded bin will more likely have more compact placement in other containers, thus, more potential for improvement.

<p style="text-align:center">
  <img src="https://latex.codecogs.com/svg.latex?\Large&space;aNB=NB%2B\frac{LeastLoad}{BinCapacity}"/>
  <center><b>Eq. 1.</b> Adjusted number of bins</center>
</p>

```python
def fitness(num_opend_bins, Bins):
    # find least load
    leastLoad = 1
    for k in range(num_opend_bins):
        load = Bins[k].load()
        if load < leastLoad:
            leastLoad = load
    
    # remainder of 1 in case 100% load 
    return self.num_opend_bins + leastLoad % 1
```

Finally, we can define the _cal_fitness_ to calculate fitness value for each solution:

```python
def cal_fitness(population)
    fitness_list = list()
    for solution in population:
        decoder = placement_procedure(BPS, VBO)
        fitness_list.append(fitness(decoder))
    return fitness_list
```
where _decoder_ is the instance of _placement_procedure_. Feel free to customerize your class for placement procedure, or compare it with mine.

<!-- # Visualization


In a instance with 100 boxes and container with shape of (600, 250, 250),  

<p style="text-align:center">
  <img src="./resource/images/history.png" />
  <center><b>Fig. 7.</b> Fitness value during GA</center>
</p>


# Conclusion
 -->
