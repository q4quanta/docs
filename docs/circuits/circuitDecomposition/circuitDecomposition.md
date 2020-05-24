### Universality: Circuit Decomposition

This notebook is from IBM Challange 2020 [GitHub](https://github.com/qiskit-community/may4_challenge_exercises)

Wow! If you managed to solve the first three exercises, congratulations! The fourth problem is supposed to puzzle even the quantum experts among you, so don’t worry if you cannot solve it. If you can, hats off to you!

You may recall from your quantum mechanics course that quantum theory is unitary. Therefore, the evolution of any (closed) system can be described by a unitary. But given an arbitrary unitary, can you actually implement it on your quantum computer?

[**"A set of quantum gates is said to be universal if any unitary transformation of the quantum data can be efficiently approximated arbitrarily well as a sequence of gates in the set."**](https://qiskit.org/textbook/ch-algorithms/defining-quantum-circuits.html)

Every gate you run on the IBM Quantum Experience is transpiled into single qubit rotations and CNOT (CX) gates. We know that these constitute a universal gate set, which implies that any unitary can be implemented using only these gates. However, in general it is not easy to find a good decomposition for an arbitrary unitary. Your task is to find such a decomposition.

You are given the following unitary:


```python
import numpy as np
from qiskit import Aer, QuantumCircuit, execute
from qiskit.visualization import plot_histogram
from IPython.display import display, Math, Latex
from may4_challenge.ex1 import return_state, vec_in_braket, statevec
from may4_challenge.ex4 import check_circuit, submit_circuit
from qiskit.compiler import transpile
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
```

#### Provided Unitary Matrix


```python
from may4_challenge.ex4 import get_unitary
U = get_unitary()
# print(U)
print("U has shape", U.shape)
```

    U has shape (16, 16)


#### Trial Zero


```python
iqc = QuantumCircuit(4)
iqc.iso(U, [0,1,2,3], [])
tiqc = transpile(iqc, basis_gates=['u3', 'cx'],optimization_level = 2)
check_circuit(tiqc)
```

    Circuit stats:
    ||U-V||_2 = 4.2634217757239e-14
    (U is the reference unitary, V is yours, and the global phase has been removed from both of them).
    Cost is 1663
    
    Something is not right with your circuit: the cost of the circuit is too high (above 1600)


#### Matrix Analysis
Lets visualize the Matrix
We can look at absolute values of matrix element.


```python
def getAbs(U):
    M = np.eye(16,16)
    for i in range(16):
        for j in range(16):
            M[i,j] = abs(U[i,j])
    return M
M = getAbs(U)
```


```python
plt.figure(figsize = [15,10])
sns.heatmap(M, annot =True, fmt = '0.2f')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fa416aef450>




![png](output_8_1.png)


#### Diagonalization of U


```python
from qiskit import BasicAer
backend = BasicAer.get_backend('unitary_simulator')
Hqc = QuantumCircuit(4)
Hqc.h(0)
Hqc.h(1)
Hqc.h(2)
Hqc.h(3)
job = execute(Hqc, backend)
H = job.result().get_unitary(Hqc, decimals=3)
H = np.matrix(H)
```


```python
DU = np.dot(np.dot(H,U),H)
ADU = getAbs(DU)
plt.figure(figsize = [10,8])
sns.heatmap(ADU, annot =True, fmt = '0.1f')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fa416bab450>




![png](output_11_1.png)


#### What circuit would make such a complicated unitary?

Is there some symmetry, or is it random? We just updated Qiskit with the introduction of a [quantum circuit library](https://github.com/Qiskit/qiskit-terra/tree/master/qiskit/circuit/library). This library gives users access to a rich set of well-studied circuit families, instances of which can be used as benchmarks (quantum volume), as building blocks in building more complex circuits (adders), or as tools to explore quantum computational advantage over classical computation (instantaneous quantum polynomial complexity circuits).

**Using only single qubit rotations and CNOT gates, find a quantum circuit that approximates that unitary \\(U \\) by a unitary \\(V \\) up to an error \\( \varepsilon = 0.01\\), such that \\(\lVert U - V\rVert_2 \leq \varepsilon\\) !** 

Note that the norm we are using here is the spectral norm, \\( \qquad \lVert A \rVert_2 = \max_{\lVert \psi \rVert_2= 1} \lVert A \psi \rVert \\).

This can be seen as the largest scaling factor that the matrix $A$ has on any initial (normalized) state $\psi$. One can show that this norm corresponds to the largest singular value of $A$, i.e., the square root of the largest eigenvalue of the matrix $A^\dagger A$, where $A^{\dagger}$ denotes the conjugate transpose of $A$.

**When you submit a circuit, we remove the global phase of the corresponding unitary $V$ before comparing it with $U$ using the spectral norm. For example, if you submit a circuit that generates $V = \text{e}^{i\theta}U$, we remove the global phase $\text{e}^{i\theta}$ from $V$ before computing the norm, and you will have a successful submission. As a result, you do not have to worry about matching the desired unitary, $U$, up to a global phase.**

As the single-qubit gates have a much higher fidelity than the two-qubit gates, we will look at the number of CNOT-gates, \\( n_{cx} \\), and the number of u3-gates, \\( n_{u3}\\), to determine the cost of your decomposition as 

$$
\qquad \text{cost} = 10 \cdot n_{cx} + n_{u3}
$$

Try to optimize the cost of your decomposition. 

**Note that you will need to ensure that your circuit is composed only of \\(u3\\) and \\(cx\\) gates. The exercise is considered correctly solved if your cost is smaller than 1600.**

---
For useful tips to complete this exercise as well as pointers for communicating with other participants and asking questions, please take a look at the following [repository](https://github.com/qiskit-community/may4_challenge_exercises). You will also find a copy of these exercises, so feel free to edit and experiment with these notebooks.

---


```python
from qiskit import BasicAer
backend = BasicAer.get_backend('unitary_simulator')
Hqc = QuantumCircuit(4)
Hqc.h(0)
Hqc.h(1)
Hqc.h(2)
Hqc.h(3)
job = execute(Hqc, backend)
H = job.result().get_unitary(Hqc, decimals=3)
H = np.matrix(H)
```


```python
NU = np.dot(U,H)
#NU = np.dot(H,U)
```


```python
##### build your quantum circuit here
qc = QuantumCircuit(4)
qc.h(0)
qc.h(1)
qc.h(2)
qc.h(3)
qc.iso(NU, [0,1,2,3], [])
#qc.h(0)
#qc.h(1)
#qc.h(2)
#qc.h(3)
#qc.draw(output='mpl') # we draw the circuit
```




    <qiskit.circuit.instructionset.InstructionSet at 0x7fa484630250>




```python
#qc = transpile(qc, basis_gates=['u3', 'cx'],optimization_level = 1)
tqc = transpile(qc, basis_gates=['u3', 'cx'],optimization_level = 2)
#qc.draw(output='mpl') # we draw the circuit
```


```python
##### check your quantum circuit by running the next line
check_circuit(tqc)
```

    Circuit stats:
    ||U-V||_2 = 7.431794894034265e-15
    (U is the reference unitary, V is yours, and the global phase has been removed from both of them).
    Cost is 331
    
    Great! Your circuit meets all the constrains.
    Your score is 331. The lower, the better!
    Feel free to submit your answer and remember you can re-submit a new circuit at any time!


You can check whether your circuit is valid before submitting it with `check_circuit(qc)`. Once you have a valid solution, please submit it by running the following cell (delete the `#` before `submit_circuit`). You can re-submit at any time.



```python
#Send the circuit as the final answer, can re-submit at any time
#submit_circuit(tqc) 
```



<div style="border: 2px solid black; padding: 2rem;">
    <p>
        Success 🎉! Your circuit has been submitted. Return to the
        <a href="https://quantum-computing.ibm.com/challenges/4anniversary/?exercise=4" target="_blank">
            IBM Quantum Challenge page
        </a>
        and check your score and ranking.
    </p>
    <p>
        Remember that you can submit a circuit as many times as you
        want.
    </p>
</div>


#### Visualize the Circuit


```python
tqc.draw(output='mpl') # we draw the circuit
```




![png](output_22_0.png)



#### More Exact Solution


```python
qc = QuantumCircuit(4)

qc.h(range(4))
qc.unitary(Uprime, qc.qubits)
qc.h(range(4))
qc = transpile(qc, basis_gates = ['u3', 'cx'], optimization_level=3)
check_circuit(qc)
qc.draw()
# apply operations to your quantum circuit here
```

Circuit stats:
||U-V||_2 = 5.470975623515672e-15
(U is the reference unitary, V is yours, and the global phase has been removed from both of them).
Cost is 149

Great! Your circuit meets all the constrains.
Your score is 149. The lower, the better!
Feel free to submit your answer and remember you can re-submit a new circuit at any time!

![final](final.png)

#### References

1. https://qiskit.org/documentation/stubs/qiskit.compiler.transpile.html
2. https://qiskit.org/documentation/stubs/qiskit.circuit.QuantumCircuit.iso.html
3. https://github.com/Qiskit/qiskit-tutorials/blob/master/legacy_tutorials/terra/5_using_the_transpiler.ipynb
4. https://github.com/Qiskit/qiskit-tutorials/blob/master/tutorials/terra/advanced/4_transpiler_passes_and_passmanager.ipynb
5. https://arxiv.org/pdf/1501.06911.pdf


```python

```
