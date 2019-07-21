
# Getting Started with Qiskit

Here, we provide an overview of working with Qiskit. Qiskit provides the basic building blocks necessary to program quantum computers. The fundamental unit of Qiskit is the **quantum circuit**. A workflow using Qiskit consists of two stages: **Build** and **Execute**. **Build** allows you to make different quantum circuits that represent the problem you are solving, and **Execute** allows you to run them on different backends.  After the jobs have been run, the data is collected. There are methods for putting this data together, depending on the program. This either gives you the answer you wanted, or allows you to make a better program for the next instance.


```python
import numpy as np
from qiskit import *
%matplotlib inline
```

## Circuit Basics <a id='circuit_basics'></a>


### Building the circuit

The basic elements needed for your first program are the QuantumCircuit, and QuantumRegister.


```python
# Create a Quantum Register with 3 qubits.
q = QuantumRegister(3, 'q')

# Create a Quantum Circuit acting on the q register
circ = QuantumCircuit(q)
```

<div class="alert alert-block alert-info">
<b>Note:</b> Naming the QuantumRegister is optional and not required.
</div>

After you create the circuit with its registers, you can add gates ("operations") to manipulate the registers. As you proceed through the tutorials you will find more gates and circuits; below is an example of a quantum circuit that makes a three-qubit GHZ state

$$|\psi\rangle = \left(|000\rangle+|111\rangle\right)/\sqrt{2}.$$

To create such a state, we start with a three-qubit quantum register. By default, each qubit in the register is initialized to\\(|0\rangle\\)To make the GHZ state, we apply the following gates:
* A Hadamard gate\\(H\\)on qubit 0, which puts it into a superposition state.
* A controlled-Not operation ($C_{X}$) between qubit 0 and qubit 1.
* A controlled-Not operation between qubit 0 and qubit 2.

On an ideal quantum computer, the state produced by running this circuit would be the GHZ state above.

In Qiskit, operations can be added to the circuit one by one, as shown below.


```python
# Add a H gate on qubit 0, putting this qubit in superposition.
circ.h(q[0])
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
# the qubits in a Bell state.
circ.cx(q[0], q[1])
# Add a CX (CNOT) gate on control qubit 0 and target qubit 2, putting
# the qubits in a GHZ state.
circ.cx(q[0], q[2])
```




    <qiskit.extensions.standard.cx.CnotGate at 0x122bbc1d0>



## Visualize Circuit

You can visualize your circuit using Qiskit `QuantumCircuit.draw()`, which plots the circuit in the form found in many textbooks.


```python
circ.draw()
```




![png](output_8_0.png)



In this circuit, the qubits are put in order, with qubit zero at the top and qubit two at the bottom. The circuit is read left to right (meaning that gates that are applied earlier in the circuit show up further to the left).

<div class="alert alert-block alert-info">
<b>Note:</b> If you don't have matplotlib set up as your default in '~/.qiskit/settings.conf' it will use a text-based drawer over matplotlib. To set the default to matplotlib, use the following in the settings.conf
    
    [default]
    circuit_drawer = mpl
    
For those that want the full LaTeX experience, you can also set the circuit_drawer = latex.

</div>

-------------

## Simulating circuits using Qiskit Aer <a id='aer_simulation'></a>

Qiskit Aer is our package for simulating quantum circuits. It provides many different backends for doing a simulation. Here we use the basic Python version.

### Statevector backend

The most common backend in Qiskit Aer is the `statevector_simulator`. This simulator returns the quantum 
state, which is a complex vector of dimensions\\(2^n\\), where \\(n\\) is the number of qubits 
(so be careful using this as it will quickly get too large to run on your machine).

<div class="alert alert-block alert-info">


When representing the state of a multi-qubit system, the tensor order used in Qiskit is different than that used in most physics textbooks. Suppose there are \\( n \\) qubits, and qubit \\(j\\) is labeled as \\(Q_{j}\\) Qiskit uses an ordering in which the \\(n^{\mathrm{th}} \\) qubit is on the <em><strong>left</strong></em> side of the tensor product, so that the basis vectors are labeled as \\(Q_n\otimes \cdots  \otimes  Q_1\otimes Q_0\\).

For example, if qubit zero is in state 0, qubit 1 is in state 0, and qubit 2 is in state 1, Qiskit would represent this state as\\(|100\rangle$, whereas many physics textbooks would represent it as\\(|001\rangle\\).

This difference in labeling affects the way multi-qubit operations are represented as matrices. For example, Qiskit represents a controlled-X (\\( C_{X}\\) operation with qubit 0 being the control and qubit 1 being the target as

$$C_X = \begin{pmatrix} 1 & 0 & 0 & 0 \\\  0 & 0 & 0 & 1 \\\ 0 & 0 & 1 & 0 \\\ 0 & 1 & 0 & 0 \\\ \end{pmatrix}.$$

</div>

To run the above circuit using the statevector simulator, first you need to import Aer and then set the backend to `statevector_simulator`.


```python
# Import Aer
from qiskit import BasicAer

# Run the quantum circuit on a statevector simulator backend
backend = BasicAer.get_backend('statevector_simulator')
```

Now that we have chosen the backend, it's time to compile and run the quantum circuit. In Qiskit we provide the `execute` function for this. ``execute`` returns a ``job`` object that encapsulates information about the job submitted to the backend.


<div class="alert alert-block alert-info">
<b>Tip:</b> You can obtain the above parameters in Jupyter. Simply place the text cursor on a function and press Shift+Tab.
</div>


```python
# Create a Quantum Program for execution 
job = execute(circ, backend)
```

When you run a program, a job object is made that has the following two useful methods: 
`job.status()` and `job.result()`, which return the status of the job and a result object, respectively.

<div class="alert alert-block alert-info">
<b>Note:</b> Jobs run asynchronously, but when the result method is called, it switches to synchronous and waits for it to finish before moving on to another task.
</div>


```python
result = job.result()
```

The results object contains the data and Qiskit provides the method 
`result.get_statevector(circ)` to return the state vector for the quantum circuit.


```python
outputstate = result.get_statevector(circ, decimals=3)
print(outputstate)
```

    [0.707+0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j
     0.707+0.j]


Qiskit also provides a visualization toolbox to allow you to view these results.

Below, we use the visualization function to plot the real and imaginary components of the state density matrix \rho.


```python
from qiskit.visualization import plot_state_city
plot_state_city(outputstate)
```




![png](output_20_0.png)



### Unitary backend

Qiskit Aer also includes a `unitary_simulator` that works _provided all the elements in the circuit are unitary operations_. This backend calculates the \\(2^n \times 2^n\\) matrix representing the gates in the quantum circuit. 


```python
# Run the quantum circuit on a unitary simulator backend
backend = BasicAer.get_backend('unitary_simulator')
job = execute(circ, backend)
result = job.result()

# Show the results
print(result.get_unitary(circ, decimals=3))
```

    [[ 0.707+0.j  0.707+0.j  0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j
       0.   +0.j  0.   +0.j]
     [ 0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j
       0.707+0.j -0.707+0.j]
     [ 0.   +0.j  0.   +0.j  0.707+0.j  0.707+0.j  0.   +0.j  0.   +0.j
       0.   +0.j  0.   +0.j]
     [ 0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j  0.707+0.j -0.707+0.j
       0.   +0.j  0.   +0.j]
     [ 0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j  0.707+0.j  0.707+0.j
       0.   +0.j  0.   +0.j]
     [ 0.   +0.j  0.   +0.j  0.707+0.j -0.707+0.j  0.   +0.j  0.   +0.j
       0.   +0.j  0.   +0.j]
     [ 0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j
       0.707+0.j  0.707+0.j]
     [ 0.707+0.j -0.707+0.j  0.   +0.j  0.   +0.j  0.   +0.j  0.   +0.j
       0.   +0.j  0.   +0.j]]


### OpenQASM backend

The simulators above are useful because they provide information about the state output by the ideal circuit and the matrix representation of the circuit. However, a real experiment terminates by _measuring_ each qubit (usually in the computational\\(|0\rangle, |1\rangle\\)basis). Without measurement, we cannot gain information about the state. Measurements cause the quantum system to collapse into classical bits. 

For example, suppose we make independent measurements on each qubit of the three-qubit GHZ state
$$|\psi\rangle = |000\rangle +|111\rangle)/\sqrt{2},$$
and let\\(xyz\\)denote the bitstring that results. Recall that, under the qubit labeling used by Qiskit,\\(x\\)would correspond to the outcome on qubit 2,\\(y\\)to the outcome on qubit 1, and\\(z\\)to the outcome on qubit 0. 

<div class="alert alert-block alert-info">
<b>Note:</b> This representation of the bitstring puts the most significant bit (MSB) on the left, and the least significant bit (LSB) on the right. This is the standard ordering of binary bitstrings. We order the qubits in the same way, which is why Qiskit uses a non-standard tensor product order.
</div>

Recall the probability of obtaining outcome\\(xyz\\)is given by
$$\mathrm{Pr}(xyz) = |\langle xyz | \psi \rangle |^{2}$\\)and as such for the GHZ state probability of obtaining 000 or 111 are both 1/2.

To simulate a circuit that includes measurement, we need to add measurements to the original circuit above, and use a different Aer backend.


```python
# Create a Classical Register with 3 bits.
c = ClassicalRegister(3, 'c')
# Create a Quantum Circuit
meas = QuantumCircuit(q, c)
meas.barrier(q)
# map the quantum measurement to the classical bits
meas.measure(q,c)

# The Qiskit circuit object supports composition using
# the addition operator.
qc = circ+meas

#drawing the circuit
qc.draw()
```




![png](output_26_0.png)



This circuit adds a classical register, and three measurements that are used to map the outcome of qubits to the classical bits. 

To simulate this circuit, we use the ``qasm_simulator`` in Qiskit Aer. Each run of this circuit will yield either the bitstring 000 or 111. To build up statistics about the distribution of the bitstrings (to, e.g., estimate\\(\mathrm{Pr}(000)$), we need to repeat the circuit many times. The number of times the circuit is repeated can be specified in the ``execute`` function, via the ``shots`` keyword.


```python
# Use Aer's qasm_simulator
backend_sim = BasicAer.get_backend('qasm_simulator')

# Execute the circuit on the qasm simulator.
# We've set the number of repeats of the circuit
# to be 1024, which is the default.
job_sim = execute(qc, backend_sim, shots=1024)

# Grab the results from the job.
result_sim = job_sim.result()
```

Once you have a result object, you can access the counts via the function `get_counts(circuit)`. This gives you the _aggregated_ binary outcomes of the circuit you submitted.


```python
counts = result_sim.get_counts(qc)
print(counts)
```

    {'000': 506, '111': 518}


Approximately 50 percent of the time, the output bitstring is 000. Qiskit also provides a function `plot_histogram`, which allows you to view the outcomes. 


```python
from qiskit.visualization import plot_histogram
plot_histogram(counts)
```




![png](output_32_0.png)



The estimated outcome probabilities\\(\mathrm{Pr}(000)\\)and \\(\mathrm{Pr}(111)\\)are computed by taking the aggregate counts and dividing by the number of shots (times the circuit was repeated). Try changing the ``shots`` keyword in the ``execute`` function and see how the estimated probabilities change.

## Running circuits using the IBM Q provider <a id='ibmq_provider'></a>

To faciliate access to real quantum computing hardware, we have provided a simple API interface.
To access IBM Q devices, you'll need an API token. For the public IBM Q devices, you can generate an API token [here](https://quantumexperience.ng.bluemix.net/qx/account/advanced) (create an account if you don't already have one). For Q Network devices, login to the q-console, click your hub, group, and project, and expand "Get Access" to generate your API token and access url.

Our IBM Q provider lets you run your circuit on real devices or on our HPC simulator. Currently, this provider exists within Qiskit, and can be imported as shown below. For details on the provider, see [The IBM Q Provider](2_the_ibmq_provider.ipynb).


```python
from qiskit import IBMQ
```

After generating your API token, call: `IBMQ.save_account('MY_TOKEN')`. For Q Network users, you'll also need to include your access url: `IBMQ.save_account('MY_TOKEN', 'URL')`

This will store your IBM Q credentials in a local file.  Unless your registration information has changed, you only need to do this once.  You may now load your accounts by calling,


```python
IBMQ.load_accounts(hub=None)
```

Once your account has been loaded, you can view the list of backends available to you.


```python
print("Available backends:")
IBMQ.backends()
```

    Available backends:
    [<IBMQBackend('ibmqx4') from IBMQ()>,
     <IBMQBackend('ibmqx2') from IBMQ()>,
     <IBMQBackend('ibmq_16_melbourne') from IBMQ()>,
     <IBMQSimulator('ibmq_qasm_simulator') from IBMQ()>]



### Running circuits on real devices

Today's quantum information processors are small and noisy, but are advancing at a fast pace. They provide a great opportunity to explore what [noisy, intermediate-scale quantum (NISQ)](https://arxiv.org/abs/1801.00862) computers can do.

The IBM Q provider uses a queue to allocate the devices to users. We now choose a device with the least busy queue that can support our program (has at least 3 qubits).


```python
from qiskit.providers.ibmq import least_busy

large_enough_devices = IBMQ.backends(filters=lambda x: \
          x.configuration().n_qubits < 10 and
         not x.configuration().simulator)
backend = least_busy(large_enough_devices)
print("The best backend is " + backend.name())
```

    The best backend is ibmqx4


To run the circuit on the backend, we need to specify the number of shots and the number of credits we are willing to spend to run the circuit. Then, we execute the circuit on the backend using the ``execute`` function.


```python
from qiskit.tools.monitor import job_monitor
# Number of shots to run the program (experiment); maximum is 8192 shots.
# Maximum number of credits to spend on executions. 
shots = 1024          
max_credits = 3        

job_exp = execute(qc, backend=backend, \
          shots=shots, max_credits=max_credits)
job_monitor(job_exp)
```

    Job Status: job has successfully run


``job_exp`` has a ``.result()`` method that lets us get the results from running our circuit.

<div class="alert alert-block alert-info">
<b>Note:</b> When the .result() method is called, the code block will wait until the job has finished before releasing the cell.
</div>


```python
result_exp = job_exp.result()
```

Like before, the counts from the execution can be obtained using ```get_counts(qc)``` 


```python
counts_exp = result_exp.get_counts(qc)
plot_histogram([counts_exp,counts])
```




![png](output_48_0.png)



### Simulating circuits using a HPC simulator

The IBM Q provider also comes with a remote optimized simulator called ``ibmq_qasm_simulator``. This remote simulator is capable of simulating up to 32 qubits. It can be used the 
same way as the remote real backends. 


```python
simulator_backend = IBMQ.get_backend('ibmq_qasm_simulator', hub=None)
```


```python
# Number of shots to run the program (experiment); maximum is 8192 shots.
# Maximum number of credits to spend on executions. 
shots = 1024           
max_credits = 3       

job_hpc = execute(qc, backend=simulator_backend,\
             shots=shots, max_credits=max_credits)
```


```python
result_hpc = job_hpc.result()
```


```python
counts_hpc = result_hpc.get_counts(qc)
plot_histogram(counts_hpc)
```




![png](output_53_0.png)



### Retrieving a previously run job

If your experiment takes longer to run then you have time to wait around, or if you simply want to retrieve old jobs, the IBM Q backends allow you to do that.
First you would need to note your job's ID:


```python
jobID = job_exp.job_id()

print('JOB ID: {}'.format(jobID))        
```

    JOB ID: 5ccf7d36557a5600718c5793


Given a job ID, that job object can be later reconstructed from the backend using retrieve_job:


```python
job_get=backend.retrieve_job(jobID)
```

and then the results can be obtained from the new job object. 


```python
job_get.result().get_counts(qc)
```




    {'000': 445,
     '001': 13,
     '010': 23,
     '111': 340,
     '110': 70,
     '100': 31,
     '101': 71,
     '011': 31}




```python

```
