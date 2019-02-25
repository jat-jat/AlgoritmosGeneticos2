# -*- coding: utf-8 -*-
"""
Este módulo contiene una implementación del agoritmo genético orientado a encontrar un camino/recorrido
en un entorno cuadriculado, sin colisiones; para un robot.
"""
__author__ = "Argüello Tello y Pérez Sarmiento"
__copyright__ = "Copyright 2019, UPChiapas"
__docformat__ = 'reStructuredText'

import numpy as np
from random import randint
import scipy.stats # Contiene la función CDR, para calcular la probabilidad acumulada de cruce por individuo.
import matplotlib.pyplot as plt # Para crear gráficas.
from matplotlib.patches import Rectangle
import matplotlib.ticker as plticker
from matplotlib.collections import PatchCollection
import json
from Interseccion import closed_segment_intersect as intersectan

class Individuo:
    """
    Representa a un individuo.
    """
    def __init__(self, calculadora, escenario:dict = None, cromosoma:list = None):
        """
        Constructor.
        Si está creando una población inicial, envíe un escenario y no envíe un cromosoma.
        El individuo se creará aleatoriamente.
        Si cruzó dos individuos, envíe el cromosoma del hijo en el respectivo parámetro.
        :param calculador: Objeto para calcular el fenotipo y el fitness.
        :param escenario: Diccionario/json con la información del escenario y el recorrido.
        :param cromosoma: (Opcional) Un cromosoma: una lista de números enteros.
        """
        self.__calc = calculadora

        if cromosoma is None:
            nuevo_cromosoma = []
            for punto in range(0, escenario['recorrido']['puntos_intermedios']):
                nuevo_cromosoma.append(randint(0, escenario['maximos']['x']))
                nuevo_cromosoma.append(randint(0, escenario['maximos']['y']))
            self.set_cromosoma(nuevo_cromosoma)
        else:
            # Se usa el cromosoma del parámetro.
            self.set_cromosoma(cromosoma)

    def get_cromosoma(self):
        """
        Getter del cromosoma o genotipo.
        :return: Cadena actual del cromosoma.
        """
        return self.__cromosoma

    def set_cromosoma(self, crom):
        """
        Setter del cromosoma/genotipo; el cual, actualiza el valor de fitness automáticamente.
        :param crom: El nuevo valor del atributo.
        """
        self.__cromosoma = crom
        self.__fitness = self.__calc.calcular_fitness(self.__cromosoma)

    def get_fitness(self):
        """
        Getter del fitness. No tiene un setter, porque se calcula automática al llamar el método 'set_cromosoma'.
        :return: float El valor de fitness actual de este individuo.
        """
        return self.__fitness

    def get_fenotipo(self):
        """
        Retorna el recorrido que este individuo representa.
        :return: El fenotipo de este individuo.
        """
        return self.__calc.get_fenotipo(self.__cromosoma)

    def reproducirse(self, pareja):
        """
        Hace que este individuo se cruce con otro.
        :param pareja:
        :return: list Lista con 2 hijos (instancias de la clase 'Individuo').
        """
        # Punto de corte.
        pt_corte = randint(1, len(self.__cromosoma) - 1)

        # Cromosomas de los hijos
        cromosomas_hijos = [[], []]
        # Cromosomas de los padres
        cromosomas_padres = [self.__cromosoma, pareja.get_cromosoma()]

        # Se crean a los hijos.
        cromosomas_hijos[0] += cromosomas_padres[0][:pt_corte] + cromosomas_padres[1][pt_corte:]
        cromosomas_hijos[1] += cromosomas_padres[1][:pt_corte] + cromosomas_padres[0][pt_corte:]

        # Se crean los hijos, usando los cromosomas armados.
        hijos = []
        hijos.append(Individuo(self.__calc, cromosoma=cromosomas_hijos[0]))
        hijos.append(Individuo(self.__calc, cromosoma=cromosomas_hijos[1]))

        return hijos

    def __str__(self):
        """
        Devuelve la información de este individuo, de modo que sea fácil de entender por el usuario.
        :return: str Cadena que representa este objeto.
        """
        return self.__repr__()

    def __repr__(self):
        # Se obtiene el recorrido, de tal modo que el usuario lo entienda fácilmente.
        recorrido = ''
        fenotipo = self.get_fenotipo()
        for i in range(0, len(fenotipo) - 2, 2):
            recorrido += '({!s}, {!s}) 🡆 '.format(int(fenotipo[i] - 0.5), int(fenotipo[i + 1] - 0.5))
        recorrido += '({!s}, {!s})'.format(int(fenotipo[-2] - 0.5), int(fenotipo[-1] - 0.5))

        return "Cromosoma/Genotipo: {!s}\nFitness: {!s}\n\nFenotipo:\n{!s}" \
               "\nRecorrido:\n{!s}"\
            .format(self.__cromosoma, self.__fitness, fenotipo, recorrido)


class CalculadoraIndividuo:
    """
    Objeto que permite calcular los fenotipos y los valores de fitness de los individuos.
    Todos los individuos de una población debe tener una referencia a la misma calculadora.
    """
    def __init__(self, escenario:dict):
        """
        Constructor.
        :param escenario: Diccionario/json con la información del escenario y el recorrido.
        """
        # Los segmentos de todos los obstáculos en el plano cartesiano.
        seg_obstaculos = []
        for obst in escenario['obstaculos']:
            # Un obstáculo es un rectángulo. Así que se guardan los 4 segmentos/lados que lo componen.
            # Lado inferior.
            seg_obstaculos.append([[obst['x'], obst['y']], [obst['x'] + obst['anchura'], obst['y']]])
            # Lado izquierdo.
            seg_obstaculos.append([[obst['x'] + obst['anchura'], obst['y']],
                                   [obst['x'] + obst['anchura'], obst['y'] + obst['altura']]])
            # Lado superior.
            seg_obstaculos.append([[obst['x'] + obst['anchura'], obst['y'] + obst['altura']],
                                   [obst['x'], obst['y'] + obst['altura']]])
            # Lado derecho.
            seg_obstaculos.append([[obst['x'], obst['y'] + obst['altura']], [obst['x'], obst['y']]])

        # Se guardan los segmentos para evaluar colisiones.
        self.__seg_obstaculos = seg_obstaculos

        ''' Se calcula y guarda la penalización (valor añadido al fitness por cada segmento del camino que colisione).
            Es el fitness correspondiente a recorrer todo el escenario. '''
        self.__penalizacion = escenario['tamano']['anchura'] * escenario['tamano']['altura']
        
        # Se guardan los puntos de inicio y fin del recorrido, como listas de dos posiciones, cada una.
        self.__pto_inicio = [escenario['recorrido']['inicio']['x'], escenario['recorrido']['inicio']['y']]
        self.__pto_fin = [escenario['recorrido']['fin']['x'], escenario['recorrido']['fin']['y']]

    def contar_colisiones(self, a:list, b:list):
        """
        Cuenta con cuántos de los segmentos que forman los obstáculos del escenario, el segmento (a, b) se intersecta.
        :param a: Un punto expresado como una lista de dos posiciones. Ejemplo: [1.5, 2.5].
        :param b: Un punto expresado como una lista de dos posiciones. Ejemplo: [1.5, 2.5].
        :return: La cantidad de veces que el camino (a, b) queda obstaculizado.
        """
        colisiones = 0
        for seg_obstaculo in self.__seg_obstaculos:
            if intersectan(a, b, seg_obstaculo[0], seg_obstaculo[1]):
                colisiones += 1
        return colisiones

    def get_penalizacion(self):
        return self.__penalizacion

    def get_fenotipo(self, cromosoma):
        """
        Devuelve el recorrido que representa cierto cromosoma/genotipo, como una lista donde cada par de
        elementos representa a un punto del recorrido.
        Los puntos están ubicados en el centro de sus respectivos cuadros (de la cuadrícula que representa
        al escenario), es por ello que todas las coordenadas terminan en '.5'.
        :param cromosoma: Cromosoma del individuo del que se quiere conocer su fenotipo.
        :return: Lista con las coordenadas de todos los puntos del recorrido.
        """
        recorrido = self.__pto_inicio + cromosoma + self.__pto_fin
        # Se centran las coordenadas.
        for i in range(len(recorrido)):
            recorrido[i] += 0.5
        return recorrido

    def calcular_fitness(self, cromosoma:list):
        """
        Calcula el valor de fitness de un individuo.
        :param cromosoma: El cromosoma del inviduo a evaluar.
        :return: float Valor de fitness.
        """
        fitness = 0
        recorrido = self.get_fenotipo(cromosoma)

        # En cada iteración de este ciclo, se procesa un segmento/parte del recorrido.
        for i in range(0, len(recorrido) - 2, 2):
            ''' Recuerde que cada pareja de elementos del recorrido, es un punto P(x,y).
                Se procesa el segmento entre estos puntos:
                (recorrido[i], recorrido[i + 1]) y (recorrido[i + 2], recorrido[i + 3]). '''
            # Se agrega la longitud del segmento al fitness.
            fitness += abs(recorrido[i + 2] - recorrido[i]) + abs(recorrido[i + 3] - recorrido[i + 1])
            # Se penaliza al fitness por cada colisión que tenga con los segmentos de los obstáculos.
            fitness += (self.__penalizacion *
                        self.contar_colisiones([recorrido[i], recorrido[i + 1]], [recorrido[i + 2], recorrido[i + 3]]))

        return fitness


class PoblacionAlgGen:
    """
    Clase que representa a una población, capaz de realizar las operaciones de un algoritmo genético.
    El objetivo del algoritmo, es hallar el camino más corto para un robot en una cuadrícula, de tal modo que no se
    choque con ningún obstáculo.
    """
    def __init__(self, ruta_escenario:str, tam_pob_ini:int, prob_mut_ind:float, prob_mut_gen:float, tam_pob_max:float=100):
        """
        Constructor.
        :param ruta_escenario: La ruta de un archivo .json, que contenga la información del escenario
        (tamaño y obstáculos), así como la del recorrido del robot (cantidad de puntos intermedios, puntos inicial y
        punto final). Puede crear su propio archivo guiándose del ejemplo adjunto a la aplicación.
        :param tam_pob_ini: Tamaño de la población inicial.
        :param prob_mut_ind: Probabilidad de mutación por individuo.
        :param prob_mut_gen: Probabilidad de mutación por gen.
        :param tam_pob_max: El tamaño máximo de la población.
        """
        if (prob_mut_ind <= 0.0) or (prob_mut_ind > 1.0) or (prob_mut_gen <= 0.0) or (prob_mut_gen > 1.0):
            raise ValueError("Las probabilidades deben de estar en el rango '(0, 1]'.")
        elif tam_pob_ini < 1:
            raise ValueError('Tamaño de la población inicial inválido.')

        # Se guardan las probabilidades de mutación, en el formato de numpy.
        self.prob_mut_ind = [1 - prob_mut_ind, prob_mut_ind]
        self.prob_mut_gen = [1 - prob_mut_gen, prob_mut_gen]
        # Se guarda la cantidad máxima de individuos.
        self.tam_pob_max = tam_pob_max

        # Se guarda la información del escenario, para poder graficarlo después.
        with open(ruta_escenario, "r") as f:
            self.escenario = json.load(f)
            self.escenario['maximos'] = {'x': self.escenario['tamano']['anchura'] - 1,
                                         'y': self.escenario['tamano']['altura'] - 1}

        # Se crea la calculadora para los individuos.
        calculadora_ind = CalculadoraIndividuo(self.escenario)
        # Se guarda la penalización, para corroborar si un recorrido está libre de colisiones.
        self.penalizacion = calculadora_ind.get_penalizacion()

        # Se crea la población inicial.
        self.poblacion = []
        for i in range(tam_pob_ini):
            self.poblacion.append(Individuo(calculadora_ind, escenario=self.escenario))

    def cruzar(self):
        """
        Efectúa una temporada de cruza entre los individuos de la población, del modo 'todos contra todos'.
        Aumenta el tamaño de la población.
        """
        # Se ordena la población por su fitness.
        self.poblacion.sort(key=lambda individuo: individuo.get_fitness(), reverse=True)

        # Se extraen los valores de fitness.
        fitnesses = [individuo.get_fitness() for individuo in self.poblacion]
        # Se obtiene la probabilidad de cruce por cada individuo.
        if(min(fitnesses) == max(fitnesses)):
            # Todos los fitnesses son iguales. Así que todos los individuos tendrán la misma probabilidad.
            prob_cruce = np.full(len(fitnesses), 0.5)
        else:
            # Se calcula la media y la desviación estándar.
            ''' Se obtiene, usando la probabilidad acumulada en una distribución normal. 
                Las probabilidades se invierten (si una es 0.2, pasa a ser 0.8) porque mientras más
                pequeño se un fitness, es más probable que se reproduzca. '''
            prob_cruce = 1 - scipy.stats.norm.cdf(fitnesses, np.mean(fitnesses), np.std(fitnesses))

        for i in range(1, len(self.poblacion)):
            # Se evalúa si el individuo actual se va a cruzar.
            if prob_cruce[i] <= np.random.random():
                # El individuo actual busca una o varias parejas, con valores de fitness menores al suyo.
                for j in range(0, i):
                    # Se evalúa si se tendrá hijo con esta pareja.
                    if prob_cruce[j] <= np.random.random():
                        # Los individuos con los índices 'i' y 'j', tienen dos hijos.
                        hijos = self.poblacion[i].reproducirse(self.poblacion[j])
                        # Los hijos pasan por el proceso de mutación,
                        self.mutar(hijos)
                        # Los hijos pasan a ser parte de la población.
                        self.poblacion.extend(hijos)

    def mutar(self, hijos):
        """
        Muta a individuos recién nacidos (cambia algunos de sus genes de '0' a '1' y viceversa) al azar.
        """
        for individuo in hijos:
            # Se evalúa si este individuo va a mutar.
            if np.random.choice(2, p = self.prob_mut_ind):
                # Cromosoma mutado.
                crom_mutado = individuo.get_cromosoma()
                # Se itera por cada gen/bit.
                for i_gen in range(len(crom_mutado)):
                    # Se evalúa si este gen va a mutar.
                    if np.random.choice(2, p=self.prob_mut_gen):
                        crom_mutado[i_gen] = randint(0, (self.escenario['maximos']['x'] if i_gen % 2 == 0
                                                         else self.escenario['maximos']['y']))
                # Se cambia el cromosoma actual por el mutado.
                individuo.set_cromosoma(crom_mutado)

    def podar(self):
        """
        Se eliminan a los individuos menos aptos de la población.
        Se usa uno de estos criterior:
            - Por una población máxima.
            - Eliminar 1/3 de los menos aptos.
        """
        # Se ordena la población por su fitness.
        self.poblacion.sort(key=lambda individuo: individuo.get_fitness(), reverse=True)

        # INICIO PODA
        if len(self.poblacion) > self.tam_pob_max:
            self.poblacion = self.poblacion[(-1 * self.tam_pob_max):]
        else:
            # Se elimina a un tercio de los individuos con peor fitness.
            self.poblacion = self.poblacion[int(len(self.poblacion)/3):]
    
    def actualizar_salida_grafica(self, minimos, maximos, promedios):
        """
        Actualiza, con la información de la generación actual, las listas necesarias para generar la
        segunda salida descrita en el método 'evolucionar'.

        :param minimos: Arreglo con los valores de fitness mínimos de las generaciones pasadas.
        :param maximos: Arreglo con los valores de fitness máximos de las generaciones pasadas.
        :param promedios: Arreglo con los valores promedio de fitness de las generaciones pasadas.
        """
        # 1.- AÑADIR LA INFORMACIÓN DE FITNESS DE ESTA GENERACIÓN
        # Se extraen los valores de fitness de la población actual.
        fitnesses = [individuo.get_fitness() for individuo in self.poblacion]
        # Se guardan los valores estadísticos de los fitnesses de esta iteración.
        minimos.append(min(fitnesses))
        maximos.append(max(fitnesses))
        promedios.append(np.mean(fitnesses))

    def evolucionar(self, iteraciones:int, porcentaje_conv:float=1.0, retornar_gen = False):
        """
        Hace que la población evolucione, es decir, que se avance cierta cantidad de generaciones.

        Además del valor retornato, este método da las siguientes salidas:
        1.- Una gráfica donde se visualiza la evolución de la media, el mejor caso y el peor caso,
        de los valores de fitness, para todas las generaciones, antes de realizar la poda.
        2.- Una imagen donde se aprecia el escenario y el recorrido del mejor individuo de la última generación.

        :param iteraciones: Por cuántas generaciones se va a avanzar.
        :param porcentaje_conv: En qué porcentaje deben converger los mejores individuos de una generación,
        para cancelar el resto de las iteraciones.
        :param retornar_gen: (Opcional) Si es True, la cantidad de iteraciones que se llevaron a cabo y
        no se graficará nada.
        :return: La información del individuo más apto de la última generación.
        """
        if iteraciones < 1:
            raise ValueError('Valor de iteraciones erróneo.')
        elif not (0.0 < porcentaje_conv <= 1.0):
            raise ValueError('El porcentaje de convergencia debe estar en el rango (0,1].')

        # Valores de fitness a graficar por cada iteración.
        minimos, maximos, promedios = [], [], []

        # Se guardan los datos estadísticos de la generación inicial.
        self.actualizar_salida_grafica(minimos, maximos, promedios)

        # Las generaciones avanzan.
        for i in range(1, iteraciones + 1):

            self.cruzar() # 'mutar' es mandado a llamar por 'cruzar'.
            
            self.actualizar_salida_grafica(minimos, maximos, promedios)
            
            self.podar()

            # Se extraen los valores de fitness.
            fitnesses = [individuo.get_fitness() for individuo in self.poblacion]
            ''' Si al menos, el porcentaje de individuos, especificado en 'porcentaje_conv',
                tienen el mejor valor de fitness (que no tenga ningún obstáculo/penalización);
                se considera que ya se ha convergido a la solución óptima,
                por lo que ya no es necesario seguir iterando. 
                Este chequeo se realiza, una vez, se haya llegado a la polbación máxima.'''
            if((len(fitnesses) == self.tam_pob_max) and
                    (min(fitnesses) < self.penalizacion) and
                    ((fitnesses.count(min(fitnesses)) / len(fitnesses)) >= porcentaje_conv)):
                break

            #print('Generación: {!s} - Población: {!s}'.format(i, len(fitnesses)))

        if retornar_gen:
            return i

        # El mejor individuo de la generación más reciente.
        mejor_ind = self.poblacion[-1]

        '''
            SE GRAFICAN LOS DATOS ESTADÍSTICOS DE LOS FITNESS.
        '''
        plt.clf() # Se eliminan gráficas pasadas.
        # Se crea el arraglo con los valores del eje x.
        indices_iteraciones = np.arange(0, i + 1)
        # Se grafican los valores máximos, mínimos y los promedios.
        plt.plot(indices_iteraciones, maximos, lw=2, label='Peor caso')
        plt.plot(indices_iteraciones, minimos, lw=2, label='Mejor caso')
        plt.plot(indices_iteraciones, promedios, lw=2, label='Promedio')
        plt.legend()
        # Se agregan las etiquetas de los ejes y el título.
        plt.xlabel('Generación/Iteración')
        plt.ylabel('Fitness')
        plt.title('Evolución de los valores de fitness (antes de la poda)')
        # Se muestra una cuadrícula.
        plt.grid(True)

        '''
            SE GRAFICA EL ESCENARIO CON EL CAMINO DEL MEJOR INDIVIDUO DE LA ÚLTIMA GENERACIÓN.
        '''
        # Fenotipo del mejor individuo, el recorrido representado por él.
        fen_mejor_ind = mejor_ind.get_fenotipo()
        # Las coordenadas de los puntos, separadas por ejes.
        x_camino, y_camino = [], []
        for i in range(0, len(fen_mejor_ind), 2):
            x_camino.append(fen_mejor_ind[i])
            y_camino.append(fen_mejor_ind[i + 1])

        # Los obstáculos, como rectángulos, para ser graficados.
        rec_obstaculos = []
        for obst in self.escenario['obstaculos']:
            # Se guarda el obstáculo, como un rectángulo que puede ser graficado por Matplotlib.
            rec_obstaculos.append(Rectangle((obst['x'], obst['y']), obst['anchura'], obst['altura']))
        fig, ax = plt.subplots()
        # Se grafican los obstáculos.
        pc = PatchCollection(rec_obstaculos, facecolor='black')
        ax.add_collection(pc)
        # Se grafica el camino del mejor individuo.
        ax.plot(x_camino, y_camino, marker='o')
        ax.plot(x_camino[0], y_camino[0], 'gX', label='Inicio')
        ax.plot(x_camino[-1], y_camino[-1], 'rX', label='Fin')
        # Se limitan los ejes.
        plt.xlim(0, self.escenario['tamano']['anchura'])
        plt.ylim(0, self.escenario['tamano']['altura'])
        # Se dibuja la cuadrícula.
        loc = plticker.MultipleLocator(base=1)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)
        # Se titula la gráfica.
        ax.set_title('Recorrido del mejor individuo de la última generación')
        # Se muestra la gráfica.
        plt.grid(True)
        plt.legend()
        plt.show()

        # Se devuelve la información del individuo más apto.
        return str(mejor_ind)