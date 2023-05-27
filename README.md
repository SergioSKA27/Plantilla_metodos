# Plantilla_metodos


Plantilla para proyecto de metodos numericos II

Contiene algunos metodos programados con graficación incluida solo falta completar los faltantes(BFGS, Integracion, Derivacion), tambien puedes agregar tus propios metodo o modificar cualquier parte del codigo.

## Graficacion con plotly para graficar una funcion crea una lista con los valores de $x,y$ y $z$ segun sea el caso crea un objeto con 

plo = gro.Figure()

posteriormente añade un trazo con 

plo.add_trace(gro.Scatter(x=valores de x,y=valores de y))


usa gro.Scatter para graficar figuras en 2 dimensiones. Para graficar figuras en 3 dimensiones crea un array con las coordenadas de x,y,z 

coordenadas = [[x_1,y_1,z_1],---,[x_n,y_n,z_n]]

y añadade un trazo a la figura plo con la funcion gro.Surface

 plo.add_trace(gro.Surface(z=coordenadas))



