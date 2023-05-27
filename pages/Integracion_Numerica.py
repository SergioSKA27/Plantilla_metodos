import streamlit as st
import pandas as pd
import numpy as np
import sympy as sy
from matplotlib import pyplot as plt
from sympy.plotting.plot import MatplotlibBackend, Plot
from sympy.plotting import plot3d,plot3d_parametric_line
import plotly as ply
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




st.title(':blue[Integración Numérica)]')

st.subheader(':blue[Descripción del método]')

st.subheader(':blue[Ejemplo]')
st.subheader('Método')
xxs = st.text_input('Ingrese la función $f(x)$: ',value='exp(-x)')



fx = sy.parse_expr(xxs,transformations='all')
intstrr = ''


fxxs = st.text_input('Ingrese el intervalo $[a,b]$: ',value='[0,1]')


for t in fxxs:

    if t != '{' and t != '}' and t != '[' and t != ']' and t != '(' and t != ')' and t != ' ':
        intstrr = intstrr + t

interval = list(map(float,intstrr.split(',')))

symbs = list(fx.free_symbols)

dxx = ''
integ = sy.integrate(fx, symbs)
for i in symbs:
    dxx = dxx +'d'+ str(i)

st.latex(r'\int '+sy.latex(fx)+dxx+' = '+sy.latex(integ)+' + C')

st.latex(r'\int_{'+str(interval[0])+'}^{'+str(interval[1])+'}'+sy.latex(fx)+dxx)

if len(fx.free_symbols)<= 2:
    if len(fx.free_symbols) == 1:
        func = sy.lambdify(list(fx.free_symbols),fx)
        plo = gro.Figure()
        plo.add_trace(gro.Scatter(x=np.linspace(interval[0],interval[1],1000),y=func(np.linspace(interval[0],interval[1],1000)),fill='tozeroy'))
        plo.update_layout(title='Integral de f'+str(tuple(fx.free_symbols))+' en el intervalo '+str(interval))
        st.plotly_chart(plo)


    if  len(fx.free_symbols) == 2:
        func = sy.lambdify(list(fx.free_symbols),fx)
        plo = gro.Figure()
        ran = np.linspace(-10,10,100)
        su = [[func(ran[xs],ran[ys]) for xs in range (0,len(ran)) ] for ys in range(0,len(ran))]
        plo.add_trace(gro.Surface(z=su))
        st.plotly_chart(plo)
        p =plot3d(fx,show=False)
        pl = get_sympy_subplots(p)

        st.pyplot(pl)


#COLOCA TU METODO AQUI y PASA LA  FUNCION ALOJADA EN fx
