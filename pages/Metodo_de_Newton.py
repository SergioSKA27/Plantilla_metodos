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


def parse_inputsys(inp):
    eq = []
    sfix = ''

    for i in inp:
        if i != '{' and i != '}' and i != '[' and i != ']' and i != '(' and i != ')' and i != ' ':
            sfix = sfix + i

    for t in sfix.split(','):
        eq.append(sy.parse_expr(t,transformations='all'))

    return eq

def get_maxsymbs(sys):
    maxs = 0
    s = []
    for i in sys:
        if max(maxs,len(i.free_symbols)) != maxs:
            s = list(i.free_symbols)

    return s

def jacobian(ff,symb):
    """
    It takes a vector of functions and a vector of symbols and returns the Jacobian matrix of the functions with respect to
    the symbols
    :param ff: the function
    :param symb: the symbols that are used in the function
    :return: A matrix of the partial derivatives of the function with respect to the variables.
    """
    m = []

    for i in range(0,len(ff)):
        aux  = []
        for j in range(0,len(symb)):
            aux.append(sy.diff(ff[i],symb[j]))
        m.append(aux)

    return np.array(m)

def hessian(ff,symb):
    """
    It takes a vector of functions and a vector of symbols and returns the Hessian matrix of the functions with respect to
    the symbols
    :param ff: a list of functions of the form f(x,y,z)
    :param symb: the symbols that are used in the function
    :return: A matrix of the second derivatives of the function.
    """

    m = []

    for i in range(0,len(ff)):
        aux  = []
        for j in range(0,len(symb)):
            aux.append(sy.diff(ff[i],symb[j],2))
        m.append(aux)
    return np.array(m)

def eval_matrix(matrix , v,symb):
    """
    It takes a matrix, a list of symbols and a list of values, and returns the matrix with the symbols substituted by the
    values

    :param matrix: the matrix of the system of equations
    :param v: the vector of values for the variables
    :param symb: the symbols that will be used in the matrix
    :return: the matrix with the values of the variables substituted by the values of the vector v.
    """
    e = 0
    mm = []
    for i in range(0,len(matrix)):
        aux = []
        ev = []
        for k in range(0,len(symb)):
            ev.append((symb[k],v[k]))
        for j in range(len(matrix[i])):
            aux.append(matrix[i][j].subs(ev).evalf())
        mm.append(aux)
    return np.array(mm)

def evalVector(ff, x0,symb):
    """
    > Given a list of symbolic expressions, a list of values for the symbols, and a list of the symbols, evaluate the
    symbolic expressions at the given values

    :param ff: the vector of functions
    :param x0: initial guess
    :param symb: the symbols that are used in the symbolic expression
    :return: the value of the function at the point x0.
    """
    v = []
    for i in range(0,len(ff)):
        ev = []

        for k in range(0,len(x0)):
            ev.append((symb[k],x0[k]))

        v.append(ff[i].subs(ev).evalf())
    return np.array(v)

def NewtonMethod( ff, x0,symb ):
    """
    The function takes in a vector of functions, a vector of initial guesses, and a vector of symbols. It then calculates
    the Jacobian matrix, the Jacobian matrix evaluated at the initial guess, the inverse of the Jacobian matrix evaluated at
    the initial guess, the vector of functions evaluated at the initial guess, and then the Newton step.

    The function returns the Newton step.

    :param ff: the function we want to find the root of
    :param x0: initial guess
    :param symb: the symbols used in the function
    :return: The return value is the x_np1 value.
    """
    j = jacobian(ff,symb)
    #print("Jacobian Matrix")
    #pprint(Matrix(j))
    jev = sy.Matrix( eval_matrix(j,x0,symb))
    #print("J(",x0,")")
    #pprint(jev)

    jinv = jev.inv()
    #print("F(",x0,")")
    ffev = sy.Matrix(evalVector(np.transpose(ff),x0,symb))
    #print("J^-1(",x0,")*","F(",x0,")")
    mm = sy.Matrix(jinv)*ffev
    #pprint(mm)
    x_np1 = sy.Matrix(np.transpose(np.array(x0)))
    #pprint(x_np1-mm)
    return list(x_np1-mm)

def norm_inf(x_0,x_1):
    """
    > The function `norm_inf` takes two vectors `x_0` and `x_1` and returns the maximum absolute difference between the two
    vectors

    :param x_0: the initial guess
    :param x_1: the vector of the current iteration
    :return: The maximum difference between the two vectors.
    """
    a = [abs(x_1[i]-x_0[i]) for i in range(len(x_0))]
    return max(a)

def newton_method(ff,x_0,symbs,error,maxiter):
    """
    Given a function (x,y)$, a starting point $, and a list of symbols,
    the function will return the next point $ in the Newton's method sequence

    :param ff: the function to be minimized
    :param x_0: initial guess
    :param symbs: the symbols that we're using in the function
    :return: the final value of x_0, the list of x values, and the list of y values.
    """
    #pprint(Matrix(x_0))
    xs = []
    ys = []
    xns = [x_0]
    erros = []
    iterr = 0
    while True and iterr < maxiter:

        x_1 = NewtonMethod(ff,x_0,symbs)
        #print(x_1)
        ninf = norm_inf(x_0,x_1)
        erros.append(ninf)
        #print(ninf)

        x_0 = list(x_1)
        xns.append(tuple(x_0))
        xs.append(x_0[0])
        ys.append(x_0[1])
        if ninf < error:
            #print("Iteraciones: ",iterr)
            break
        iterr = iterr+1

    #print(x_0)
    return xns,xs,ys,erros


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


st.title(':blue[Metodo de Newton]')

st.subheader(':blue[Descripción del método]')

methd_desc1 = r'''
    El método de Newton para sistemas de ecuaciones no lineales es una extensión del método utilizado para
resolver una sola ecuación, por tanto sigue la misma estrategia que se empleó en el caso de una sola ecuación:
linealizar y resolver, repitiendo los pasos con la frecuencia necesaria. Consideramos, por sencillez, el caso de
dos ecuaciones con dos variables:
'''

methd_desc2 = r'''

\begin{cases}
f(x,y) = 0 \\
g(x,y) = 0 \\
\end{cases}
'''

methd_desc3 = r'''
F'(x,y) = \begin{pmatrix}
\frac{\partial f}{\partial x} (x,y) & \frac{\partial f}{\partial y} (x,y) \\
\frac{\partial g}{\partial x} (x,y) & \frac{\partial g}{\partial y} (x,y)
\end{pmatrix}
'''

methd_desc4 = r'''
\begin{pmatrix}
x_{n+1}\\
y_{n+1}
\end{pmatrix} =
\begin{pmatrix}
x_{n}\\
y_{n}
\end{pmatrix} - (F'(x_n,y_n))^{-1} F(x_n,y_n)
'''


st.write(methd_desc1)
st.latex(methd_desc2)
st.write('Si definimos la función $F(x, y) = (f (x, y), g(x, y))$, transformamos el sistema anterior en')
st.latex('F(x,y) = 0')
st.write(''' Podemos realizar entonces una formulación idéntica al caso de una variable donde la derivada está dada en
este caso por la matriz jacobian''')

st.latex(methd_desc3)

st.write('''Así, el método de Newton comienza por un vector inicial $(x_0, y_0)^T$ y calcula el resto de aproximaciones
mediante ''')

st.latex(methd_desc4)

st.write('''
donde $(F'(x_n, y_n))^1$ es la matriz inversa de $F'(x_n, y_n)$. Para poder aplicar este método es necesario que
$F'(x, y)$ sea no singular.\\
Un problema del método anterior es que el cálculo de la matriz inversa es costoso computacionalmente y
debemos calcularla en cada paso. Esto se puede resolver descomponiendo el método en dos etapas:
''')

st.write('1. Resolver el sistema lineal con dos ecuaciones y dos incognitas:')
st.latex(r'''
F'(x_n, y_n)
\begin{pmatrix}
u_{n}\\
v_{n}
\end{pmatrix} = -F(x_n, y_n)
''')

st.write('2. Tomar como nueva aproximación:')
st.latex(r'''
\begin{pmatrix}
x_{n+1}\\
y_{n+1}
\end{pmatrix} =
\begin{pmatrix}
x_{n}\\
y_{n}
\end{pmatrix} +
\begin{pmatrix}
u_{n}\\
v_{n}
\end{pmatrix}
''')


st.subheader(':blue[Ejemplo]')








st.subheader('Método')
sys = st.text_input('Ingrese el sistema de ecuaciones ',value=r'{x^2+3*y*x+2,y**3*x**2-2*y**3-5}')
try:
    system = parse_inputsys(sys)
except :
    st.error('Error al introducir el sistema de ecuaciones', icon="🚨")

st.write('_El sistema de ecuaciones es:_')
psys ='F'+str(tuple(get_maxsymbs(system)))+ r'''
    =\begin{cases}

'''
for i in system:
    psys = psys + sy.latex(i)
    psys = psys + r' = 0\\'

psys = psys + r'\end{cases}'

st.latex(psys)

fx = sy.lambdify(list(get_maxsymbs(system)),system[0])
fx2 = sy.lambdify(list(get_maxsymbs(system)),system[1])




try:
    st.write('_Grafica del sistema como función implicita_ : ')
    x,y = sy.symbols('x,y')
    p1 = sy.plot_implicit(system[0],(x,-10,10),(y,-10,10),show=False,line_color='red')
    for i in range(1,len(system)):
        p1.append(sy.plot_implicit(system[i],(x,-10,10),(y,-10,10),show=False,line_color='blue')[0])
    t2dplot =  get_sympy_subplots(p1)
    st.pyplot(t2dplot)

    st.write('_Grafica 3D del sistema_')

    plo1 = gro.Figure()

    ran = np.linspace(-10,10,100)
    su1 = [[fx(ran[xs],ran[ys]) for xs in range (0,len(ran)) ] for ys in range(0,len(ran))]
    su2 = [[fx2(ran[xs],ran[ys]) for xs in range (0,len(ran)) ] for ys in range(0,len(ran))]

    plo1.add_trace(gro.Surface(z=su1,name='Ecuación 1',opacity=.7))
    plo1.add_trace(gro.Surface(z=su2,name='Ecuación 2',opacity=.7))

    st.plotly_chart(plo1)
    p13d = plot3d(system[0],(x,-10,10),(y,-10,10),show=False,surface_color='r')
    for i in range(1,len(system)):
        p13d.append(plot3d(system[i],(x,-10,10),(y,-10,10),show=False,surface_color='g')[0])

    t3dplot = get_sympy_subplots(p13d)
    st.pyplot(t3dplot)

except Exception as excep:
    st.error(excep)
    st.error('Error al graficar', icon="🚨")


initaprx = st.text_input('Ingrese una aproximacion inicial $x_0$: ',value=[-1,1])

intaprox = []
intstr = ''




for i in initaprx:

    if i != '{' and i != '}' and i != '[' and i != ']' and i != ' ':
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


st.write('_Matrix Jacobiana_:')
symbs = (get_maxsymbs(system))


st.latex(sy.latex(sy.Matrix(jacobian(system,symbs))))

method = newton_method(system,list(intaprox),symbs,float(err),maxiter)

tabb = []

for i in range(0,len(method[1])):
    aux = list(method[0][i])
    aux.append(method[3][i])
    tabb.append(aux)

cols = list(map(str, list(symbs)))
cols.append('Error')
#st.write(tabb)
table = pd.DataFrame(tabb,columns=cols)

st.write(table)
try:


    #st.write(method[2])

    xs =[float(i) for i in method[1]]
    xy =[float(i) for i in method[2]]
    evalfx = [float(fx(i[0],i[1])) for i in method[0]]

    plo = gro.Figure()
    plo.add_trace(gro.Scatter3d(x=xs,y=xy,z=evalfx,name='Aproximaciones'))

    ranx = np.linspace(int(method[1][-1]-1),int(method[1][-1]+1),100)
    rany = np.linspace(int(method[2][-1]-1),int(method[2][-1]+1),100)
    su1 = [[fx(ranx[xs],rany[ys]) for xs in range (0,len(ranx)) ] for ys in range(0,len(rany))]
    su2 = [[fx2(ranx[xs],rany[ys]) for xs in range (0,len(ranx)) ] for ys in range(0,len(rany))]


    plo.add_trace(gro.Surface(z=su1,name='Ecuación 1',opacity=.8))
    plo.add_trace(gro.Surface(z=su2,name='Ecuación 2',opacity=.7))

    plo.update_layout(title='Grafica del Sistema')
    st.plotly_chart(plo)


    pf =get_sympy_subplots(p1)
    pf.plot(method[1],method[2],'o')
    st.pyplot(pf)
    auxpd = plot3d(system[0],(x,-1+method[1][-1],method[1][-1]+1),(y,-1+method[2][-1],method[2][-1]+1),show=False,alpha=0.3,title='Grafica de la Solución')
    auxpd.append(plot3d(system[1],(x,-1+method[1][-1],method[1][-1]+1),(y,-1+method[2][-1],method[2][-1]+1),show=False,alpha=0.5)[0])
    pda = get_sympy_subplots(auxpd)
    zs = []
    for i in range(0,len(method[1])):
        zs.append(system[0].subs({x : method[1][i],y: method[2][i]}).evalf())
    pda.plot(method[1],method[2],zs,'o',markersize=30,color='red')
    st.pyplot(pda)
except Exception as e:
    st.error(str(e))




