### Two Qubit Circuits


```python
%matplotlib inline
import numpy as np
import IPython
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit import BasicAer
from qiskit.tools.jupyter import *
from qiskit.visualization import *
import seaborn as sns
sns.set()
```


```python
from helper import *
import os
import glob
import moviepy.editor as mpy
```



#### Double Qubit Circuit

Base states: |00>, |01>, |10>,|11>

#### Gate on First Qubit 

 \\( I \otimes U \\)

```python
qc = QuantumCircuit(2)
qc.u3(np.pi/2,np.pi/2,np.pi/2,0)
style = {'backgroundcolor': 'lavender'}
qc.draw(output='mpl', style = style)
```




![png](output_7_0.png)




```python
getMatrix(qc)
```




    matrix([[ 0.707+0.j   , -0.   -0.707j,  0.   +0.j   ,  0.   +0.j   ],
            [ 0.   +0.707j, -0.707+0.j   ,  0.   +0.j   ,  0.   +0.j   ],
            [ 0.   +0.j   ,  0.   +0.j   ,  0.707+0.j   , -0.   -0.707j],
            [ 0.   +0.j   ,  0.   +0.j   ,  0.   +0.707j, -0.707+0.j   ]])



#### Gate on Second Qubit

\\( U \otimes I \\)

```python
qc = QuantumCircuit(2)
qc.u3(np.pi/2,np.pi/2,np.pi/2,1)
style = {'backgroundcolor': 'lavender'}
qc.draw(output='mpl', style = style)
```




![png](output_10_0.png)




```python
getMatrix(qc)
```




    matrix([[ 0.707+0.j   ,  0.   +0.j   , -0.   -0.707j,  0.   +0.j   ],
            [ 0.   +0.j   ,  0.707+0.j   ,  0.   +0.j   , -0.   -0.707j],
            [ 0.   +0.707j,  0.   +0.j   , -0.707+0.j   ,  0.   +0.j   ],
            [ 0.   +0.j   ,  0.   +0.707j,  0.   +0.j   , -0.707+0.j   ]])



####    Gate on both Qubits

 \\( U \otimes U\\)

```python
qc = QuantumCircuit(2)
qc.u3(np.pi/2,np.pi/2,np.pi/2,0)
qc.u3(np.pi/2,np.pi/2,np.pi/2,1)
style = {'backgroundcolor': 'lavender'}
qc.draw(output='mpl', style = style)
```




![png](output_13_0.png)




```python
getMatrix(qc)
```




    matrix([[ 0.5+0.j , -0. -0.5j, -0. -0.5j, -0.5+0.j ],
            [ 0. +0.5j, -0.5+0.j ,  0.5-0.j ,  0. +0.5j],
            [ 0. +0.5j,  0.5-0.j , -0.5+0.j ,  0. +0.5j],
            [-0.5+0.j , -0. -0.5j, -0. -0.5j,  0.5-0.j ]])



####   Gates in Parallel and Series 

\\( (U \times U) \otimes (U\times U) \\)

```python
qc = QuantumCircuit(2)
qc.u3(np.pi/2,np.pi/2,np.pi/2,0)
qc.u3(np.pi/2,np.pi/2,np.pi/2,0)
qc.u3(np.pi/2,np.pi/2,np.pi/2,1)
qc.u3(np.pi/2,np.pi/2,np.pi/2,1)
style = {'backgroundcolor': 'lavender'}
qc.draw(output='mpl', style = style)
```




![png](output_16_0.png)




```python
getMatrix(qc)
```




    matrix([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]])



------------------
