import numpy as np
import IPython
import ipywidgets as widgets
import colorsys
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit import execute, Aer, BasicAer
from qiskit.visualization import plot_bloch_multivector
from qiskit.tools.jupyter import *
from qiskit.visualization import *
import seaborn as sns
sns.set()


def blochSphere(qc):
    backend = BasicAer.get_backend('statevector_simulator')
    job = execute(qc,backend).result()
    return plot_bloch_multivector(job.get_statevector(qc))

def plotMatrix(qc):
    backend = BasicAer.get_backend('unitary_simulator')
    job = execute(qc, backend)
    Matrix = job.result().get_unitary(qc, decimals=3)
    M = np.matrix(Matrix)
    MD = [["o" for i in range(M.shape[0])] for j in range(M.shape[1])]
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            r = M[i,j].real
            im = M[i,j].imag
            MD[i][j] =  str(r)[0:4]+ " , " +str(im)[0:4]
    plt.figure(figsize = [2*M.shape[1],M.shape[0]])
    sns.heatmap(np.abs(M),\
                annot = np.array(MD),\
                fmt = '',linewidths=.5,\
                cmap='Blues')
    
    
    
def drawCircuit_2q(qc):
    qc.barrier()
    qc.measure([0,1],[0,1])
    style = {'backgroundcolor': 'lavender'}
    return qc.draw(output='mpl', style = style)
def drawCircuit_3q(qc):
    qc.barrier()
    qc.measure([0,1],[0,1])
    style = {'backgroundcolor': 'lavender'}
    return qc.draw(output='mpl', style = style)
def drawCircuit_4q(qc):
    qc.barrier()
    qc.measure([0,1],[0,1],[0,1],[0,1])
    style = {'backgroundcolor': 'lavender'}
    return qc.draw(output='mpl', style = style)
def drawCircuit_5q(qc):
    qc.barrier()
    qc.measure([0,1],[0,1],[0,1],[0,1],[0,1])
    style = {'backgroundcolor': 'lavender'}
    return qc.draw(output='mpl', style = style)





def drawCircuit_text(qc):
    qc.barrier()
    qc.measure([0,1], [0,1])
    style = {'backgroundcolor': 'lavender'}
    return qc.draw(output='text', style = style)


def simCircuit(qc):
    qc.barrier()
    qc.measure([0,1], [0,1])
    backend= Aer.get_backend('qasm_simulator')
    result = execute(qc,backend).result()
    counts = result.get_counts(qc)
    return plot_histogram(counts, figsize = [4,3])

def vec_in_braket(vec: np.ndarray) -> str:
    nqubits = int(np.log2(len(vec)))
    state = ''
    for i in range(len(vec)):
        rounded = round(vec[i], 3)
        if rounded != 0:
            basis = format(i, 'b').zfill(nqubits)
            state += np.str(rounded).replace('-0j', '+0j')
            state += '|' + basis + '\\rangle + '
    state = state.replace("j", "i")
    return state[0:-2].strip()
def vec_in_text_braket(vec):
    return '$$\\text{{State:\n $|\\Psi\\rangle = $}}{}$$'.format(vec_in_braket(vec))


def writeState(qc):
    backend = BasicAer.get_backend('statevector_simulator')
    job = execute(qc,backend).result()
    vec = job.get_statevector(qc)
    return widgets.HTMLMath(vec_in_text_braket(vec))

def getPhase(qc):
    backend = BasicAer.get_backend('statevector_simulator')
    job = execute(qc,backend).result()
    ket = job.get_statevector(qc)
    Phase = []
    for i in range(len(ket)):
        angles = (np.angle(ket[i]) + (np.pi * 4)) % (np.pi * 2)
        rgb = colorsys.hls_to_rgb(angles / (np.pi * 2), 0.5, 0.5)
        mag = np.abs(ket[i])
        Phase.append({"rgb":rgb,"mag": mag,"ang":angles})
    return Phase

def drawPhase(phaseDic, fsize):
    depth = len(phaseDic)
    nqubit = len(phaseDic[0])
    r = 0.30
    dx = 1.0
    d = 1.0
    plt.figure(figsize = fsize)
    for i in range(depth):
        x0 = i
        for j in range(nqubit):
            y0 = j
            mag = phaseDic[i][j]['mag']
            ang = phaseDic[i][j]['ang']
            rgb = phaseDic[i][j]['rgb']
            
            ax=plt.gca()
            circle1= plt.Circle((dx+x0,y0), radius = r, color = 'white')
            ax.add_patch(circle1)
            circle2= plt.Circle((dx+x0,y0), radius= r*mag, color = rgb)
            ax.add_patch(circle2)
            line = plt.plot((dx+x0,dx+x0+(r*mag*np.cos(ang))),(y0,y0+(r*mag*np.sin(ang))),color = "black")
            plt.yticks([x for x in range(nqubit+1)])
            #plt.axis('scaled')
            plt.xlabel("Circuit Depth")
            plt.ylabel("Basis States")
            
        
    plt.show()