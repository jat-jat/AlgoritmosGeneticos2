# -*- coding: utf-8 -*-
__author__ = "Argüello Tello y Pérez Sarmiento"
__copyright__ = "Copyright 2019, UPChiapas"
__docformat__ = 'reStructuredText'

import sys
import traceback

import AlgGen
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QInputDialog
from PyQt5 import uic, QtCore
import matplotlib.pyplot as plt

# La ruta del archivo que contiene la información de la vista del programa.
ARCHIVO_UI = "GUI.ui"

class Ventana(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        uic.loadUi(ARCHIVO_UI, self)

        # Se crea una ventana de alerta para mostrar excepciones.
        self.MSG_ERROR = QMessageBox()
        self.MSG_ERROR.setIcon(QMessageBox.Critical)
        self.MSG_ERROR.setText("Ha ocurrido una excepción:")
        self.MSG_ERROR.setStandardButtons(QMessageBox.Ok)
        self.MSG_ERROR.setBaseSize(500,300)

        # Se configuran los eventos.
        self.btn_ejecutar.clicked.connect(self.ejecutar_alg_gen)
        self.btn_pruebas_rendimiento.clicked.connect(self.prueba_rendimiento)
        self.menu_acerca_de.triggered.connect(self.acerca_de)

    def mostrar_excepcion(self, e):
        """
        Muestra una alerta con los detalles de una excepción.
        :param Exception e: Una excepción cualquiera.
        """
        self.MSG_ERROR.setWindowTitle(type(e).__name__)
        self.MSG_ERROR.setInformativeText(str(e))
        self.MSG_ERROR.setDetailedText(traceback.format_exc())
        self.MSG_ERROR.show()

    def ejecutar_alg_gen(self):
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.statusBar().showMessage("Procesando...")

        # Información del individuo más apto de la última generación.
        resStr = None
        try:
            ruta_escenario = QFileDialog.getOpenFileName(self, 'Abra el archivo con la información del escenario y recorrido',
                                                         QtCore.QDir.current().path(), "Archivo JSON (*.json)")[0]

            if ruta_escenario:
                poblacion = AlgGen.PoblacionAlgGen(ruta_escenario, self.campo_tam_pob.value(),
                                                         self.campo_prob_mut_ind.value(), self.campo_prob_mut_gen.value(),
                                                         self.campo_tam_pob_max.value())

                resStr = poblacion.evolucionar(self.campo_iteraciones.value(), self.campo_porcentaje_conv.value())
        except Exception as e:
            self.mostrar_excepcion(e)
        finally:
            self.statusBar().clearMessage()
            QApplication.restoreOverrideCursor()

        if resStr is not None:
            QMessageBox.about(self, 'Individuo más apto', resStr)

    def prueba_rendimiento(self):
        cant_ejecuciones = QInputDialog.getInt(self, "Pruebas de rendimiento", "Indique la cantidad de ejecuciones:",
                                               10, 2, 1000, 1)
        if not cant_ejecuciones[1]:
            return

        ruta_escenario = QFileDialog.getOpenFileName(self, 'Abra el archivo con la información del escenario y recorrido',
                                                     QtCore.QDir.current().path(), "Archivo JSON (*.json)")[0]
        if not ruta_escenario:
            return

        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.statusBar().showMessage("Procesando...")

        ejecuciones, generaciones_por_ejecucion = [], []
        try:
            for i in range(1, cant_ejecuciones[0] + 1):
                poblacion = AlgGen.PoblacionAlgGen(ruta_escenario, self.campo_tam_pob.value(),
                                                         self.campo_prob_mut_ind.value(), self.campo_prob_mut_gen.value(),
                                                         self.campo_tam_pob_max.value())

                generaciones = poblacion.evolucionar(self.campo_iteraciones.value(),
                                                     self.campo_porcentaje_conv.value(), retornar_gen=True)

                ejecuciones.append(i)
                generaciones_por_ejecucion.append(generaciones)

            # Se grafican los datos.
            plt.clf()  # Se eliminan gráficas pasadas.
            plt.plot(ejecuciones, generaciones_por_ejecucion)
            # Se agregan las etiquetas de los ejes y el título.
            plt.xlabel('Ejecución')
            plt.ylabel('Generaciones')
            plt.title('Prueba de rendimiento')
            # Se muestra la gráfica.
            plt.grid(True)
            plt.show()
        except Exception as e:
            self.mostrar_excepcion(e)
        finally:
            self.statusBar().clearMessage()
            QApplication.restoreOverrideCursor()


    def acerca_de(self):
        QMessageBox.about(self, "Acerca de...", "UP Chiapas\nInteligencia artificial - 8°" +
                          "\nIA.C2.A2 Algoritmos genéticos 2\n\nJavier Alberto Argüello Tello - 153217\n"
                          "Luis Alejandro Pérez Sarmiento - 163195")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    _ventana = Ventana()
    _ventana.show()
    app.exec_()
