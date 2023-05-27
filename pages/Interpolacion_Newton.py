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

def diff_div(v,fx,order):
    """
    > The function takes in a list of values, a list of function values, and an order, and returns a list of divided
    differences

    :param v: the list of x values
    :param fx: the function you want to differentiate
    :param order: the order of the derivative you want to take
    :return: the difference quotient of the function f(x)
    """

    m = []

    for i in range(0,len(fx)):
        #print(fx[i])
        if i + 1 < len(fx) and i +order < len(v):
            #print(v[i+1],v[i],"/",fx[i+order]," ",fx[i])
            m.append((fx[i+1]-fx[i])/(v[i+order]-v[i]))
    return m

def divided_diff(fx,v):
    """
    The function takes in a list of x values and a list of f(x) values, and returns a list of lists of divided differences

    :param fx: the function to be interpolated
    :param v: list of x values
    :return: The divided difference table is being returned.
    """
    x = v
    nfx = fx
    m = []
    for i in range(0,len(v)-1):
        nx = diff_div(v,nfx,i+1)
        #print(nx)
        m.append(nx)
        nfx = nx

    #print(m)
    return m

def table_divided_dif(x,fx,diff):
    table = []
    for i in range(0,len(fx)):
        table.append([x[i],fx[i]])
        table.append([' ',' '])

    for i in range(1,len(table)):
        for j in range(0,len(diff)):
            for k in range(0,len(diff[j])):
                table[i].append(diff[j][k])


    return table


def print_divdiff(v):
    """
    The function `print_divdiff` prints a divided difference table given a list of values.

    :param v: The input parameter "v" is expected to be a list of lists, where the first list contains the x values and the
    second list contains the corresponding f(x) values. The remaining lists contain the divided differences of the function
    """
    table = []
    s = ''
    for i in range(0,len(v[0])):
        table.append([str(v[0][i] ),str(v[1][i]) ])
        table.append([])
    #print(table)
    for k in range(2,len(v)):
        aux = k-1
        for j in range(0,len(v[k])):
            for t in range(0,k):
                table[aux+j].append(' ')
            table[aux+j].append( str(v[k][j]))
            aux = aux + 1
    m = 0
    for i in table:
        if len(i) > m:
            m = len(i)

    for t in table:
        while len(t) != m:
            t.append('')

    return table
def Newton_interpolation(fx,v):
    """
    It takes in a list of x values and a list of f(x) values, and returns a polynomial that interpolates the points

    :param fx: a list of the function values
    :param v: list of x values
    :return: The function is being returned.
    """
    diff = divided_diff(fx,v)
    x = sy.symbols('x')

    expr = v[0]
    expr_reg = v[0]

    for i in range(0,len(diff)):
        s = diff[i][0]
        p = 1
        for k in range(0,len(v)):

            p = p*(x-v[k])
            #print(p, "p",k)
            if k == i:
                break
        s = s * p
        expr = expr + s

    for i in range(0,len(diff)):
        s = diff[i][-1]
        p = 1
        for k in range(len(v)-1,0,-1):

            p = p*(x-v[k])
            #print(p, "p",k)
            if k == i:
                break
        s = s * p
        expr_reg = expr_reg + s

    #pprint(expr)

    p = sy.plot(expr,(x,-10,10),show=False)
    #p.append(sy.plot(expr_reg,(x,-10,10),show=False)[0])
    p2 = get_sympy_subplots(p)
    p2.plot(v,fx,"o")



    #p2.show()
    difdiv = [v,fx]
    for i in diff:
        difdiv.append(i)

    return sy.expand(expr),p2, print_divdiff(difdiv),sy.expand(expr_reg)





st.title(':blue[Interpolación de Newton]')

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

    fxxs = st.text_input('Ingrese los valores de $f(x_k)$: ',value='{1,2.14557,3.141592,4}')

    x = list(map(float,xsstr.split(',')))
    intstrr = ''




    for t in fxxs:

        if t != '{' and t != '}' and t != '[' and t != ']' and t != '(' and t != ')' and t != ' ':
            intstrr = intstrr + t

    fx = list(map(float,intstrr.split(',')))


#st.write(x)
#st.write(fx)



method = Newton_interpolation(fx,x)
try:
    st.write('Diferencias Divididas')

    st.table(method[2])
except:
    st.error('This is an error', icon="🚨")
st.write('_El polinomio de Interpolacion está dado por:_')
st.write('Progresivo: ')
st.latex('p_n(x) = ' +sy.latex(method[0]))
st.write('Regresivo:')
st.latex('p_n(x) = ' +sy.latex(method[-1]))



func = sy.lambdify(sy.symbols('x'),method[0])
funcdata = pd.DataFrame(dict(x=np.linspace(-10,10,1000),y=func(np.linspace(-10,10,1000))))
func2 = sy.lambdify(sy.symbols('x'),method[-1])
plo = gro.Figure()

plo.add_trace(gro.Scatter(x=np.linspace(-10,10,1000),y=func(np.linspace(-10,10,1000)),name='Interpolación'))
#plo.add_trace(gro.Scatter(x=np.linspace(-10,10,1000),y=func2(np.linspace(-10,10,1000)),name='Interpolación2'))
plo.add_trace(gro.Scatter(x=x,y=fx, marker_color='rgba(152, 0, 0, .8)',name='Datos'))

#plo.add_hline(y=0)
#plo.add_vline(x=0)
plo.update_layout(title='Grafica de la Interpolación')
st.plotly_chart(plo)

