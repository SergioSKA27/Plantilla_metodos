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




st.title(':blue[Método de Quasi-Newton(BFGS)]')

st.subheader(':blue[Descripción del método]')


desc = r'''
 Inicializar $x_0$ and $H_0$ \FOR{$k=0,1,2,...$}
 Calcular el vector de gradiente  $g_k$
  Determinar la dirección de búsqueda  $p_k$ resolviendo $H_kp_k = -g_k$
 Elegir el tamaño $a_k$ usando el método de búsqueda
 Actualizar el punto $x_{k+1} = x_k + a_kp_k$
 Operar el vestor $s_k = x_{k+1} - x_k$ y $y_k = g_{k+1} - g_k$
 Actualizar la aproximación del vector Hessiano usando:
H_{k+1} = (I - \rho_k s_k y_k^T)H_k(I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T

Aquí, $x_k$representa la posición actual en iteración$k$,
$g_k$  representa el vector de gradiente en $x_k$, $H_k$
representa la aproximación de la matriz de Hesse inversa en $x_k$,
$\rho_k$es el tamaño del paso para actualizar  $H_k$ en iteración  $k$,
 and $p_k$ es la dirección de búsqueda en iteration $k$.

'''
st.latex(desc)

st.subheader(':blue[Ejemplo]')

st.subheader('Método')
xxs = st.text_input('Ingrese la función $f(x)$: ',value='(x - 1)**2 + (y - 2.5)**2')



fx = sy.parse_expr(xxs)
intstrr = ''


st.latex('f'+str(tuple(fx.free_symbols))+' = '+sy.latex(fx))
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



initaprx = st.text_input('Ingrese una aproximacion inicial $x_0$: ',value='[0,0]')

intaprox = []
intstr = ''




for i in initaprx:

    if i != '{' and i != '}' and i != '[' and i != ']' and i != '(' and i != ')' and i != ' ':
        intstr = intstr + i

try:
    st.write('La aproximacion inicial es: ')
    intaprox = list(map(int, intstr.split(',')))
    st.latex(sy.latex(sy.Matrix(list(intaprox))))
except:
    st.error('Error al introducir la aproximación inicial', icon="🚨")

err = st.text_input('Ingrese el error de tolerancia: ',value='0.00001')
try:
    st.write('El error de tolerancia es:', float(err))
except:
    st.error('Error al introducir el error de tolerancia', icon="🚨")


maxiter = st.slider('Maximo de Iteraciones',10,1000,10)

