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




st.title(':blue[Derivación Numérica]')

st.subheader(':blue[Descripción del método]')

st.subheader(':blue[Ejemplo]')
st.subheader('Método')
xxs = st.text_input('Ingrese la función $f(x)$: ',value='(x - 1)**2')



fx = sy.parse_expr(xxs,transformations='all')
intstrr = ''

st.latex(r'\frac{\partial f}{\partial x}'+sy.latex(fx))


if len(fx.free_symbols)<= 2:
    if len(fx.free_symbols) == 1:
        func = sy.lambdify(list(fx.free_symbols),fx)
        plo = gro.Figure()
        plo.add_trace(gro.Scatter(x=np.linspace(-10,10,1000),y=func(np.linspace(-10,10,1000))))
        st.plotly_chart(plo)
        p =sy.plot(fx,show=False)
        pl = get_sympy_subplots(p)

        st.pyplot(pl)

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
