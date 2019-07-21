
# Qiskit Aqua: Variational Quantum Eigensolver

**Experimenting with Max-Cut problem and Traveling Salesman problem with variational quantum eigensolver**

The latest version of this notebook is available [here](https://github.com/Qiskit/qiskit-tutorial).

***
### Contributors
Antonio Mezzacapo<sup>[1]</sup>, Jay Gambetta<sup>[1]</sup>, Kristan Temme<sup>[1]</sup>, Ramis Movassagh<sup>[1]</sup>, Albert Frisch<sup>[1]</sup>, Takashi Imamichi<sup>[1]</sup>, Giacomo Nannicni<sup>[1]</sup>, Richard Chen<sup>[1]</sup>, Marco Pistoia<sup>[1]</sup>, Stephen Wood<sup>[1]</sup>
### Affiliation
- <sup>[1]</sup>IBMQ

## Introduction

Many problems in quantitative fields such as finance and engineering are optimization problems. Optimization problems lay at the core of complex decision-making and definition of strategies. 

Optimization (or combinatorial optimization) means searching for an optimal solution in a finite or countably infinite set of potential solutions. Optimality is defined with respect to some criterion function, which is to be minimized or maximized. This is typically called cost function or objective function. 

**Typical optimization problems**

Minimization: cost, distance, length of a traversal, weight, processing time, material, energy consumption, number of objects

Maximization: profit, value, output, return, yield, utility, efficiency, capacity, number of objects 

We consider here max-cut problem of practical interest in many fields, and show how they can mapped on quantum computers.


### Weighted Max-Cut

Max-Cut is an NP-complete problem, with applications in clustering, network science, and statistical physics. To grasp how practical applications are mapped into given Max-Cut instances, consider a system of many people that can interact and influence each other. Individuals can be represented by vertices of a graph, and their interactions seen as pairwise connections between vertices of the graph, or edges. With this representation in mind, it is easy to model typical marketing problems. For example, suppose that it is assumed that individuals will influence each other's buying decisions, and knowledge is given about how strong they will influence each other. The influence can be modeled by weights assigned on each edge of the graph. It is possible then to predict the outcome of a marketing strategy in which products are offered for free to some individuals, and then ask which is the optimal subset of individuals that should get the free products, in order to maximize revenues.

The formal definition of this problem is the following:

Consider an \\(n\\)-node undirected graph *G = (V, E)* where *|V| = n* with edge weights \\(w_{ij}>0, w_{ij}=w_{ji}\\), for \\((i, j)\in E\\). A cut is defined as a partition of the original set V into two subsets. The cost function to be optimized is in this case the sum of weights of edges connecting points in the two different subsets, *crossing* the cut. By assigning \\(x_i=0\\) or \\( x_i=1 \\) to each node \\(i\\), one tries to maximize the global profit function (here and in the following summations run over indices 0,1,...n-1)

$$ \tilde{C}(\textbf{x}) = \sum_{i,j} w_{ij} x_i (1-x_j). $$

In our simple marketing model, \\(w_{ij}\\) represents the probability that the person \\(j\\) will buy a product after \\(i\\) gets a free one. Note that the weights \\(w_{ij}\\) can in principle be greater than \\(1\\), corresponding to the case where the individual \\(j\\) will buy more than one product. Maximizing the total buying probability corresponds to maximizing the total future revenues. In the case where the profit probability will be greater than the cost of the initial free samples, the strategy is a convenient one. An extension to this model has the nodes themselves carry weights, which can be regarded, in our marketing model, as the likelihood that a person granted with a free sample of the product will buy it again in the future. With this additional information in our model, the objective function to maximize becomes 

$$ C(\textbf{x}) = \sum_{i,j} w_{ij} x_i (1-x_j)+\sum_i w_i x_i.$$
 
In order to find a solution to this problem on a quantum computer, one needs first to map it to an Ising Hamiltonian. This can be done with the assignment \\(x_i\rightarrow (1-Z_i)/2\\), where \\(Z_i\\) is the Pauli Z operator that has eigenvalues \\( \pm 1 \\). Doing this we find that 

$$C(\textbf{Z}) = \sum_{i,j} \frac{w_{ij}}{4} (1-Z_i)(1+Z_j) + \sum_i \frac{w_i}{2} (1-Z_i)$$

$$ = -\frac{1}{2}\left( \sum_{i<j} w_{ij} Z_i Z_j +\sum_i w_i Z_i\right)+\mathrm{const},$$


where const = \\(\sum_{i<j}w_{ij}/2+\sum_i w_i/2 \\). In other terms, the weighted Max-Cut problem is equivalent to minimizing the Ising Hamiltonian 

$$H = \sum_i w_i Z_i + \sum_{i<j} w_{ij} Z_iZ_j.$$

Aqua can generate the Ising Hamiltonian for the first profit function\\(\tilde{C}$.


### Approximate Universal Quantum Computing for Optimization Problems

There has been a considerable amount of interest in recent times about the use of quantum computers to find a solution to combinatorial problems. It is important to say that, given the classical nature of combinatorial problems, exponential speedup in using quantum computers compared to the best classical algorithms is not guaranteed. However, due to the nature and importance of the target problems, it is worth investigating heuristic approaches on a quantum computer that could indeed speed up some problem instances. Here we demonstrate an approach that is based on the Quantum Approximate Optimization Algorithm by Farhi, Goldstone, and Gutman (2014). We frame the algorithm in the context of *approximate quantum computing*, given its heuristic nature. 

The Algorithm works as follows:

-  Choose the \\(w_i\\) and \\(w_{ij}\\) in the target Ising problem. In principle, even higher powers of Z are allowed.
-  Choose the depth of the quantum circuit \\(m\\). Note that the depth can be modified adaptively.
-  Choose a set of controls \\(\theta\\) and make a trial function \\(|\psi(\boldsymbol\theta)\rangle\\), built using a quantum circuit made of C-Phase gates and single-qubit Y rotations, parameterized by the components of \\(\boldsymbol\theta\\). 
-  Evaluate 
$$ C(\boldsymbol\theta) = \langle\psi(\boldsymbol\theta)~|H|~\psi(\boldsymbol\theta)\rangle$$
$$ = \sum_i w_i \langle\psi(\boldsymbol\theta)~|Z_i|~\psi(\boldsymbol\theta)\rangle+ \sum_{i<j} w_{ij} \langle\psi(\boldsymbol\theta)~|Z_iZ_j|~\psi(\boldsymbol\theta)\rangle$$
 by sampling the outcome of the circuit in the Z-basis and adding the expectation values of the individual Ising terms together. In general, different control points around \\(\boldsymbol\theta\\) have to be estimated, depending on the classical optimizer chosen. 
- Use a classical optimizer to choose a new set of controls.
- Continue until \\(C(\boldsymbol\theta)\\) reaches a minimum, close enough to the solution \\(\boldsymbol\theta^*\\).
- Use the last \\(\boldsymbol\theta\\) to generate a final set of samples from the distribution \\(|\langle z_i~|\psi(\boldsymbol\theta)\rangle|^2\;\forall i\\) to obtain the answer.
    
It is our belief the difficulty of finding good heuristic algorithms will come down to the choice of an appropriate trial wavefunction. For example, one could consider a trial function whose entanglement best aligns with the target problem, or simply make the amount of entanglement a variable. In this tutorial, we will consider a simple trial function of the form

$$|\psi(\theta)\rangle  = [U_\mathrm{single}(\boldsymbol\theta) U_\mathrm{entangler}]^m |+\rangle$$

where\\(U_\mathrm{entangler}\\)is a collection of C-Phase gates (fully entangling gates), and\\(U_\mathrm{single}(\theta) = \prod_{i=1}^n Y(\theta_{i})\\), where\\(n\\)is the number of qubits and\\(m\\)is the depth of the quantum circuit. The motivation for this choice is that for these classical problems this choice allows us to search over the space of quantum states that have only real coefficients, still exploiting the entanglement to potentially converge faster to the solution.

One advantage of using this sampling method compared to adiabatic approaches is that the target Ising Hamiltonian does not have to be implemented directly on hardware, allowing this algorithm not to be limited to the connectivity of the device. Furthermore, higher-order terms in the cost function, such as\\(Z_iZ_jZ_k\\), can also be sampled efficiently, whereas in adiabatic or annealing approaches they are generally impractical to deal with. 

-----------

References:

-  A. Lucas, Frontiers in Physics 2, 5 (2014)
-  E. Farhi, J. Goldstone, S. Gutmann e-print arXiv 1411.4028 (2014)
-  D. Wecker, M. B. Hastings, M. Troyer Phys. Rev. A 94, 022309 (2016)
-  E. Farhi, J. Goldstone, S. Gutmann, H. Neven e-print arXiv 1703.06199 (2017)

-------------

```python
# useful additional packages 
import matplotlib.pyplot as plt
import matplotlib.axes as axes
%matplotlib inline
import numpy as np
import networkx as nx

from qiskit import BasicAer
from qiskit.tools.visualization import plot_histogram
from qiskit.aqua import Operator, run_algorithm
from qiskit.aqua.input import EnergyInput
from qiskit.aqua.translators.ising import max_cut, tsp
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.components.variational_forms import RY
from qiskit.aqua import QuantumInstance

# setup aqua logging
import logging
from qiskit.aqua import set_qiskit_aqua_logging
# set_qiskit_aqua_logging(logging.DEBUG)  
# choose INFO, DEBUG to see the log
```

### [Optional] Setup token to run the experiment on a real device
If you would like to run the experiement on a real device, you need to setup your account first.

Note: If you do not store your token yet, use `IBMQ.save_accounts()` to store it first.


```python
from qiskit import IBMQ
# IBMQ.load_accounts()
```

## Max-Cut problem

###### Generating a graph of 4 nodes 

```python


n=4 # Number of nodes in graph
G=nx.Graph()
G.add_nodes_from(np.arange(0,n,1))
elist=[(0,1,1.0),(0,2,1.0),(0,3,1.0),(1,2,1.0),(2,3,1.0)]
# tuple is (i,j,weight) where (i,j) is the edge
G.add_weighted_edges_from(elist)

colors = ['r' for node in G.nodes()]
pos = nx.spring_layout(G)
default_axes = plt.axes(frameon=True)
nx.draw_networkx(G, node_color=colors, node_size=600, alpha=.8, ax=default_axes, pos=pos)
```

![png](output_7_1.png)

---------------------

######  Computing the weight matrix from the random graph

```python

w = np.zeros([n,n])
for i in range(n):
    for j in range(n):
        temp = G.get_edge_data(i,j,default=0)
        if temp != 0:
            w[i,j] = temp['weight'] 
print(w)
```

    [[0. 1. 1. 1.]
     [1. 0. 1. 0.]
     [1. 1. 0. 1.]
     [1. 0. 1. 0.]]

----------------

### Brute force approach

Try all possible\\(2^n\\)combinations. For\\(n = 4\\), as in this example, one deals with only 16 combinations, but for n = 1000, one has 1.071509e+30 combinations, which is impractical to deal with by using a brute force approach. 


```python
best_cost_brute = 0
for b in range(2**n):
    x = [int(t) for t in reversed(list(bin(b)[2:].zfill(n)))]
    cost = 0
    for i in range(n):
        for j in range(n):
            cost = cost + w[i,j]*x[i]*(1-x[j])
    if best_cost_brute < cost:
        best_cost_brute = cost
        xbest_brute = x 
    print('case = ' + str(x)+ ' cost = ' + str(cost))

colors = ['r' if xbest_brute[i] == 0 else 'b' for i in range(n)]
nx.draw_networkx(G, node_color=colors,\
                 node_size=600, alpha=.8, pos=pos)
print('\nBest solution = ' + str(xbest_brute) +\
             ' cost = ' + str(best_cost_brute))    
```

    case = [0, 0, 0, 0] cost = 0.0
    case = [1, 0, 0, 0] cost = 3.0
    case = [0, 1, 0, 0] cost = 2.0
    case = [1, 1, 0, 0] cost = 3.0
    case = [0, 0, 1, 0] cost = 3.0
    case = [1, 0, 1, 0] cost = 4.0
    case = [0, 1, 1, 0] cost = 3.0
    case = [1, 1, 1, 0] cost = 2.0
    case = [0, 0, 0, 1] cost = 2.0
    case = [1, 0, 0, 1] cost = 3.0
    case = [0, 1, 0, 1] cost = 4.0
    case = [1, 1, 0, 1] cost = 3.0
    case = [0, 0, 1, 1] cost = 3.0
    case = [1, 0, 1, 1] cost = 2.0
    case = [0, 1, 1, 1] cost = 3.0
    case = [1, 1, 1, 1] cost = 0.0
    
    Best solution = [1, 0, 1, 0] cost = 4.0



![png](output_10_1.png)

---------------

### Mapping to the Ising problem


```python
qubitOp, offset = max_cut.get_max_cut_qubitops(w)
algo_input = EnergyInput(qubitOp)
```

### [Optional] Using DOcplex for mapping to the Ising problem
Using ```docplex.get_qubitops``` is a different way to create an Ising Hamiltonian of Max-Cut. ```docplex.get_qubitops``` can create a corresponding Ising Hamiltonian from an optimization model of Max-Cut. An example of using ```docplex.get_qubitops``` is as below. 


```python
from docplex.mp.model import Model
from qiskit.aqua.translators.ising import docplex

# Create an instance of a model and variables.
mdl = Model(name='max_cut')
x = {i: mdl.binary_var(name='x_{0}'.format(i)) for i in range(n)}

# Object function
max_cut_func = mdl.sum(w[i,j]* x[i] * ( 1 - x[j] )\
             for i in range(n) for j in range(n))
mdl.maximize(max_cut_func)

# No constraints for Max-Cut problems.
```


```python
qubitOp_docplex, offset_docplex = docplex.get_qubitops(mdl)
```

### Checking that the full Hamiltonian gives the right cost 


```python
# Making the Hamiltonian in its full form and 
# getting the lowest eigenvalue and eigenvector
ee = ExactEigensolver(qubitOp, k=1)
result = ee.run()

"""
algorithm_cfg = {
    'name': 'ExactEigensolver',
}

params = {
    'problem': {'name': 'ising'},
    'algorithm': algorithm_cfg
}
result = run_algorithm(params,algo_input)
"""
x = max_cut.sample_most_likely(result['eigvecs'][0])
print('energy:', result['energy'])
print('max-cut objective:', result['energy'] + offset)
print('solution:', max_cut.get_graph_solution(x))
print('solution objective:', max_cut.max_cut_value(x, w))

colors = ['r' if max_cut.get_graph_solution(x)[i] == 0 \
                    else 'b' for i in range(n)]
nx.draw_networkx(G, node_color=colors, \
                    node_size=600, alpha = .8, pos=pos)
```

    energy: -1.5
    max-cut objective: -4.0
    solution: [0. 1. 0. 1.]
    solution objective: 4.0


![png](output_17_2.png)

-----------------

### Running it on quantum computer
We run the optimization routine using a feedback loop with a quantum computer that uses trial functions built with Y single-qubit rotations,\\(U_\mathrm{single}(\theta) = \prod_{i=1}^n Y(\theta_{i})\\), and entangler steps\\(U_\mathrm{entangler}\\).


```python
seed = 10598

spsa = SPSA(max_trials=300)
ry = RY(qubitOp.num_qubits, depth=5, entanglement='linear')
vqe = VQE(qubitOp, ry, spsa, 'matrix')

backend = BasicAer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend, \
                        seed=seed, seed_transpiler=seed)

result = vqe.run(quantum_instance)

"""declarative approach
algorithm_cfg = {
    'name': 'VQE',
    'operator_mode': 'matrix'
}

optimizer_cfg = {
    'name': 'SPSA',
    'max_trials': 300
}

var_form_cfg = {
    'name': 'RY',
    'depth': 5,
    'entanglement': 'linear'
}

params = {
    'problem': {'name': 'ising', 'random_seed': seed},
    'algorithm': algorithm_cfg,
    'optimizer': optimizer_cfg,
    'variational_form': var_form_cfg,
    'backend': {provider': 'qiskit.BasicAer',\
                     'name': 'statevector_simulator'}
}

result = run_algorithm(params, algo_input)
"""

x = max_cut.sample_most_likely(result['eigvecs'][0])
print('energy:', result['energy'])
print('time:', result['eval_time'])
print('max-cut objective:', result['energy'] + offset)
print('solution:', max_cut.get_graph_solution(x))
print('solution objective:', max_cut.max_cut_value(x, w))

colors = ['r' if max_cut.get_graph_solution(x)[i] == 0 \
                                    else 'b' for i in range(n)]
nx.draw_networkx(G, node_color=colors,\
                                    node_size=600, alpha = .8, pos=pos)
```

    energy: -1.4999760821185284
    time: 7.753880739212036
    max-cut objective: -3.999976082118528
    solution: [1. 0. 1. 0.]
    solution objective: 4.0



![png](output_19_1.png)

---------------------------

###### Run quantum algorithm with shots

```python

seed = 10598

spsa = SPSA(max_trials=300)
ry = RY(qubitOp.num_qubits, depth=5, entanglement='linear')
vqe = VQE(qubitOp, ry, spsa, 'grouped_paulis')

backend = BasicAer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=1024,\
                         seed=seed, seed_transpiler=seed)

result = vqe.run(quantum_instance)

"""declarative approach, update the param from the previous cell.
params['algorithm']['operator_mode'] = 'grouped_paulis'
params['backend']['provider'] = 'qiskit.BasicAer'
params['backend']['name'] = 'qasm_simulator'
params['backend']['shots'] = 1024
result = run_algorithm(params, algo_input)
"""

x = max_cut.sample_most_likely(result['eigvecs'][0])
print('energy:', result['energy'])
print('time:', result['eval_time'])
print('max-cut objective:', result['energy'] + offset)
print('solution:', max_cut.get_graph_solution(x))
print('solution objective:', max_cut.max_cut_value(x, w))
plot_histogram(result['eigvecs'][0])

colors = ['r' if max_cut.get_graph_solution(x)[i] == 0\
                                 else 'b' for i in range(n)]
nx.draw_networkx(G, node_color=colors, node_size=600,\
                                 alpha = .8, pos=pos)
```

    energy: -1.5
    time: 15.852537155151367
    max-cut objective: -4.0
    solution: [1 0 1 0]
    solution objective: 4.0



![png](output_20_1.png)

--------------

### [Optional] Checking that the full Hamiltonian made by ```docplex.get_qubitops```  gives the right cost


```python
# Making the Hamiltonian in its full form and getting
#  the lowest eigenvalue and eigenvector
ee = ExactEigensolver(qubitOp_docplex, k=1)
result = ee.run()

x = docplex.sample_most_likely(result['eigvecs'][0])
print('energy:', result['energy'])
print('max-cut objective:', result['energy'] + offset_docplex)
print('solution:', max_cut.get_graph_solution(x))
print('solution objective:', max_cut.max_cut_value(x, w))

colors = ['r' if max_cut.get_graph_solution(x)[i] == 0\
                                 else 'b' for i in range(n)]
nx.draw_networkx(G, node_color=colors, \
                                node_size=600, alpha = .8, pos=pos)
```

    energy: -1.5
    max-cut objective: -4.0
    solution: [0. 1. 0. 1.]
    solution objective: 4.0



![png](output_22_1.png)

----------------

## Traveling Salesman Problem

In addition to being a notorious NP-complete problem that has drawn the attention of computer scientists and mathematicians for over two centuries, the Traveling Salesman Problem (TSP) has important bearings on finance and marketing, as its name suggests. Colloquially speaking, the traveling salesman is a person that goes from city to city to sell merchandise. The objective in this case is to find the shortest path that would enable the salesman to visit all the cities and return to its hometown, i.e. the city where he started traveling. By doing this, the salesman gets to maximize potential sales in the least amount of time. 

The problem derives its importance from its "hardness" and ubiquitous equivalence to other relevant combinatorial optimization problems that arise in practice.
 
The mathematical formulation with some early analysis was proposed by W.R. Hamilton in the early 19th century. Mathematically the problem is, as in the case of Max-Cut, best abstracted in terms of graphs. The TSP on the nodes of a graph asks for the shortest *Hamiltonian cycle* that can be taken through each of the nodes. A Hamilton cycle is a closed path that uses every vertex of a graph once. The general solution is unknown and an algorithm that finds it efficiently (e.g., in polynomial time) is not expected to exist.

Find the shortest Hamiltonian cycle in a graph \\(G=(V,E)\\)with\\(n=|V|\\) nodes and distances,\\(w_{ij}\\) (distance from vertex \\(i\\)to vertex\\(j\\). A Hamiltonian cycle is described by \\(N^2\\)variables\\(x_{i,p}\\), where\\(i\\)represents the node and\\(p\\)represents its order in a prospective cycle. The decision variable takes the value 1 if the solution occurs at node\\(i\\)at time order\\(p\\). We require that every node can only appear once in the cycle, and for each time a node has to occur. This amounts to the two constraints (here and in the following, whenever not specified, the summands run over 0,1,...N-1)

$$\sum_{i} x_{i,p} = 1 ~~\forall p$$
$$\sum_{p} x_{i,p} = 1 ~~\forall i.$$

For nodes in our prospective ordering, if\\(x_{i,p}\\)and\\(x_{j,p+1}\\)are both 1, then there should be an energy penalty if\\((i,j) \notin E\\)(not connected in the graph). The form of this penalty is 

$$\sum_{i,j\notin E}\sum_{p} x_{i,p}x_{j,p+1}>0,$$

where it is assumed the boundary condition of the Hamiltonian cycle\\((p=N)\equiv (p=0)\\). However, here it will be assumed a fully connected graph and not include this term. The distance that needs to be minimized is 

$$C(\textbf{x})=\sum_{i,j}w_{ij}\sum_{p} x_{i,p}x_{j,p+1}.$$

Putting this all together in a single objective function to be minimized, we get the following:

$$C(\textbf{x})=\sum_{i,j}w_{ij}\sum_{p} x_{i,p}x_{j,p+1}+ A\sum_p\left(1- \sum_i x_{i,p}\right)^2+A\sum_i\left(1- \sum_p x_{i,p}\right)^2,$$

where\\(A\\)is a free parameter. One needs to ensure that\\(A\\)is large enough so that these constraints are respected. One way to do this is to choose\\(A\\)such that\\(A > \mathrm{max}(w_{ij})\\).

Once again, it is easy to map the problem in this form to a quantum computer, and the solution will be found by minimizing a Ising Hamiltonian. 


```python
# Generating a graph of 3 nodes
n = 3
num_qubits = n ** 2
ins = tsp.random_tsp(n)
G = nx.Graph()
G.add_nodes_from(np.arange(0, n, 1))
colors = ['r' for node in G.nodes()]
pos = {k: v for k, v in enumerate(ins.coord)}
default_axes = plt.axes(frameon=True)
nx.draw_networkx(G, node_color=colors, node_size=600, alpha=.8, ax=default_axes, pos=pos)
print('distance\n', ins.w)
```

    distance
     [[ 0. 51. 81.]
     [51.  0. 99.]
     [81. 99.  0.]]



![png](output_24_1.png)


### Brute force approach


```python
from itertools import permutations

def brute_force_tsp(w, N):
    a=list(permutations(range(1,N)))
    last_best_distance = 1e10
    for i in a:
        distance = 0
        pre_j = 0
        for j in i:
            distance = distance + w[j,pre_j]
            pre_j = j
        distance = distance + w[pre_j,0]
        order = (0,) + i
        if distance < last_best_distance:
            best_order = order
            last_best_distance = distance
            print('order = ' + str(order) + \
                        ' Distance = ' + str(distance))
    return last_best_distance, best_order
  
best_distance, best_order = brute_force_tsp(ins.w, ins.dim)
print('Best order from brute force = ' + str(best_order) +\
                 ' with total distance = ' + str(best_distance))

def draw_tsp_solution(G, order, colors, pos):
    G2 = G.copy()
    n = len(order)
    for i in range(n):
        j = (i + 1) % n
        G2.add_edge(order[i], order[j])
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G2, node_color=colors, \
                    node_size=600, alpha=.8, ax=default_axes, pos=pos)

draw_tsp_solution(G, best_order, colors, pos)
```

    order = (0, 1, 2) Distance = 231.0
    Best order from brute force = (0, 1, 2) with total distance = 231.0



![png](output_26_1.png)


### Mapping to the Ising problem


```python
qubitOp, offset = tsp.get_tsp_qubitops(ins)
algo_input = EnergyInput(qubitOp)
```

### [Optional] Using DOcplex for mapping to the Ising problem
Using ```docplex.get_qubitops``` is a different way to create an Ising Hamiltonian of TSP. ```docplex.get_qubitops``` can create a corresponding Ising Hamiltonian from an optimization model of TSP. An example of using ```docplex.get_qubitops``` is as below. 


```python
# Create an instance of a model and variables
mdl = Model(name='tsp')
x = {(i,p): mdl.binary_var(name='x_{0}_{1}'.format(i,p)) \
                for i in range(n) for p in range(n)}

# Object function
tsp_func = mdl.sum(ins.w[i,j] * x[(i,p)] * x[(j,(p+1)%n)] \
                for i in range(n) for j in range(n) for p in range(n))
mdl.minimize(tsp_func)

# Constrains
for i in range(n):
    mdl.add_constraint(mdl.sum(x[(i,p)] for p in range(n)) == 1)
for p in range(n):
    mdl.add_constraint(mdl.sum(x[(i,p)] for i in range(n)) == 1)
```


```python
qubitOp_docplex, offset_docplex = docplex.get_qubitops(mdl)
```

------------------

### Checking that the full Hamiltonian gives the right cost 


```python
# Making the Hamiltonian in its full form and getting 
# the lowest eigenvalue and eigenvector
ee = ExactEigensolver(qubitOp, k=1)
result = ee.run()

"""
algorithm_cfg = {
    'name': 'ExactEigensolver',
}

params = {
    'problem': {'name': 'ising'},
    'algorithm': algorithm_cfg
}
result = run_algorithm(params,algo_input)
"""
print('energy:', result['energy'])
print('tsp objective:', result['energy'] + offset)
x = tsp.sample_most_likely(result['eigvecs'][0])
print('feasible:', tsp.tsp_feasible(x))
z = tsp.get_tsp_solution(x)
print('solution:', z)
print('solution objective:', tsp.tsp_value(z, ins.w))
draw_tsp_solution(G, z, colors, pos)
```

    energy: -600115.5
    tsp objective: 231.0
    feasible: True
    solution: [2, 1, 0]
    solution objective: 231.0



![png](output_33_1.png)

------------------

### Running it on quantum computer
We run the optimization routine using a feedback loop with a quantum computer that uses trial functions built with Y single-qubit rotations,\\(U_\mathrm{single}(\theta) = \prod_{i=1}^n Y(\theta_{i})\\), and entangler steps\\(U_\mathrm{entangler}\\).


```python
seed = 10598

spsa = SPSA(max_trials=300)
ry = RY(qubitOp.num_qubits, depth=5, entanglement='linear')
vqe = VQE(qubitOp, ry, spsa, 'matrix')

backend = BasicAer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend, \
                        seed=seed, seed_transpiler=seed)

result = vqe.run(quantum_instance)
"""
algorithm_cfg = {
    'name': 'VQE',
    'operator_mode': 'matrix'
}

optimizer_cfg = {
    'name': 'SPSA',
    'max_trials': 300
}

var_form_cfg = {
    'name': 'RY',
    'depth': 5,
    'entanglement': 'linear'
}

params = {
    'problem': {'name': 'ising', 'random_seed': seed},
    'algorithm': algorithm_cfg,
    'optimizer': optimizer_cfg,
    'variational_form': var_form_cfg,
    'backend': {'provider': 'qiskit.BasicAer',\
                     'name': 'statevector_simulator'}
}
result = run_algorithm(parahms,algo_input)
"""
print('energy:', result['energy'])
print('time:', result['eval_time'])
#print('tsp objective:', result['energy'] + offset)
x = tsp.sample_most_likely(result['eigvecs'][0])
print('feasible:', tsp.tsp_feasible(x))
z = tsp.get_tsp_solution(x)
print('solution:', z)
print('solution objective:', tsp.tsp_value(z, ins.w))
draw_tsp_solution(G, z, colors, pos)
```

    energy: -590938.6460660461
    time: 20.48488998413086
    feasible: True
    solution: [2, 1, 0]
    solution objective: 231.0



![png](output_35_1.png)

-----------------------

```python
# run quantum algorithm with shots

seed = 10598

spsa = SPSA(max_trials=300)
ry = RY(qubitOp.num_qubits, depth=5, entanglement='linear')
vqe = VQE(qubitOp, ry, spsa, 'grouped_paulis')

backend = BasicAer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=1024,\
                             seed=seed, seed_transpiler=seed)

result = vqe.run(quantum_instance)

"""update params in the previous cell
params['algorithm']['operator_mode'] = 'grouped_paulis'
params['backend']['provider'] = 'qiskit.BasicAer'
params['backend']['name'] = 'qasm_simulator'
params['backend']['shots'] = 1024
result = run_algorithm(params,algo_input)
"""
print('energy:', result['energy'])
print('time:', result['eval_time'])
#print('tsp objective:', result['energy'] + offset)
x = tsp.sample_most_likely(result['eigvecs'][0])
print('feasible:', tsp.tsp_feasible(x))
z = tsp.get_tsp_solution(x)
print('solution:', z)
print('solution objective:', tsp.tsp_value(z, ins.w))
plot_histogram(result['eigvecs'][0])
draw_tsp_solution(G, z, colors, pos)
```

------------------


### [Optional] Checking that the full Hamiltonian made by ```docplex.get_qubitops```  gives the right cost


```python
ee = ExactEigensolver(qubitOp_docplex, k=1)
result = ee.run()

print('energy:', result['energy'])
print('tsp objective:', result['energy'] + offset_docplex)

x = docplex.sample_most_likely(result['eigvecs'][0])
print('feasible:', tsp.tsp_feasible(x))
z = tsp.get_tsp_solution(x)
print('solution:', z)
print('solution objective:', tsp.tsp_value(z, ins.w))
draw_tsp_solution(G, z, colors, pos)
```


```python

```
