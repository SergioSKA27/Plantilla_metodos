import streamlit as st
import pandas as pd
import numpy as np
import sympy as sy
from matplotlib import pyplot as plt
from sympy.plotting.plot import MatplotlibBackend, Plot
from sympy.plotting import plot3d,plot3d_parametric_line
import plotly.express as ex
import plotly.graph_objects as gro
from plotly.subplots import make_subplots


def get_sympy_subplots(plot:Plot):
    """
    It takes a plot object and returns a matplotlib figure object

    :param plot: The plot object to be rendered
    :type plot: Plot
    :return: A matplotlib figure object.
    """
    backend = MatplotlibBackend(plot)

    backend.process_series()
    backend.fig.tight_layout()
    return backend.plt

def li(v, i):
    """
    The function takes a list of numbers and an index, and returns the Lagrange interpolating polynomial for the list of
    numbers with the index'th number removed

    :param v: the list of x values
    :param i: the index of the x value you want to interpolate
    :return: the Lagrange interpolating polynomial for the given data points.
    """
    x = sy.symbols('x')

    s = 1
    st = ''
    for k in range(0,len(v)):
        if k != i:
            st = st + '((' + str(x) + '-'+ str(v[k])+')/('+str(v[i])+'-'+str(v[k])+'))'
            s = s*((x-v[k])/(v[i]-v[k]))

    return s

def Lagrange(v,fx):
    """
    It takes in a list of x values and a list of y values, and returns the Lagrange polynomial that interpolates those
    points

    :param v: list of x values
    :param fx: The function you want to interpolate
    :return: the Lagrange polynomial.
    """
    #print(v)
    #print(fx)
    lis = []
    for i in range(0,len(v)):
        lis.append(li(v,i))

    sums = 0

    for k in range(0,len(v)):
        sums = sums+(fx[k]*lis[k])

    #print(sums)

    sy.simplify(sums)

    sy.pprint(sums)

    p1 = sy.plot(sums,show=False)
    p2 = get_sympy_subplots(p1)
    p2.plot(v,fx,"o")
    #p2.show()
    return sy.expand(sums), p2,lis

st.title(':blue[Interpolación de Lagrange]')

st.subheader(':blue[Descripción del método]')

st.subheader(':blue[Ejemplo]')


st.subheader('Método')

filess = st.sidebar.file_uploader('Selecciona un archivo de prueba: ')
if filess != None:
    fi = pd.read_csv(filess)
    st.write('Los datos a interpolar son: ')
    st.write(fi)
    x = list(fi['x'])
    fx = list(fi['y'])
else:
    xxs = st.text_input('Ingrese los valores de $x_k$: ',value='{1,2,3,4}')

    xsstr = ''


    for i in xxs:

        if i != '{' and i != '}' and i != '[' and i != ']' and i != '(' and i != ')' and i != ' ':
            xsstr = xsstr + i

    fxxs = st.text_input('Ingrese los valores de $f(x_k)$: ',value='{-1,3,4,5}')

    x = list(map(float,xsstr.split(',')))
    intstrr = ''




    for t in fxxs:

        if t != '{' and t != '}' and t != '[' and t != ']' and t != '(' and t != ')' and t != ' ':
            intstrr = intstrr + t

    fx = list(map(float,intstrr.split(',')))


#st.write(x)
#st.write(fx)
#data = [x,fx]
#st.write(data)


method = Lagrange(x,fx)

st.write('_Los polinomios fundamentales de Lagrange estan dados por:_')
lli = r'''l_i(x) = \begin{cases}'''
for t in range(0,len(method[2])):
    lli = lli +'l_'+str(t)+r'='+sy.latex(sy.expand(method[2][t]))+r'\\'
lli = lli + r'\end{cases}'
st.latex(lli)
st.write('_El polinomio de Interpolacion está dado por:_')
st.latex(r'p_n(x) = \sum_{i=0}^{n} l_i(x)f(x_i)')
st.latex('p_n(x) =' + sy.latex(method[0]))

func = sy.lambdify(sy.symbols('x'),method[0])
funcdata = pd.DataFrame(dict(x=np.linspace(-10,10,1000),y=func(np.linspace(-10,10,1000))))

plo = gro.Figure()

plo.add_trace(gro.Scatter(x=np.linspace(-10,10,1000),y=func(np.linspace(-10,10,1000)),name='Interpolación'))
plo.add_trace(gro.Scatter(x=x,y=fx, marker_color='rgba(152, 0, 0, .8)',name='Datos'))
#plo.add_hline(y=0)
#plo.add_vline(x=0)
plo.update_layout(title='Grafica de la Interpolación')
st.plotly_chart(plo)

st.pyplot(method[1])
