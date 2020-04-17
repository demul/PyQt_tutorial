# system modules
import sys

# UI modules
import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui

# visualization modules
import cv2
import numpy as np


class STTApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        ##### set basic UI setting
        ## set window title
        self.setWindowTitle('STT App')
        ## set geometry
        self.setGeometry(100, 100, 1800, 700)
        ## set title bar icon
        self.data_image = np.load('dark.npy')
        img = QtGui.QPixmap(QtGui.QImage(self.data_image, self.data_image.shape[1], self.data_image.shape[0],
                                         self.data_image.shape[1] * 3, QtGui.QImage.Format_RGB888).rgbSwapped())
        self.setWindowIcon(QtGui.QIcon(img))

        ##### set widgets
        ## set reference viewer label
        self.label_reference_viewer = self.set_reference_viewer()
        ## set STT Results viewer label
        self.label_STT_result_viewer = self.set_STT_result_viewer()
        ## set Analysis Results viewer label
        self.label_analysis_result_viewer = self.set_analysis_result_viewer()
        ## set import button
        self.btn_importer = self.set_importer()
        ## set calculate button
        self.btn_calculator = self.set_calculator()
        ## set delete button
        self.btn_deleter = self.set_deleter()

        ##### set layout
        grid = QtWidgets.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.btn_importer, 0, 0, 1, 6)
        grid.addWidget(self.btn_calculator, 0, 6, 1, 6)
        grid.addWidget(self.btn_deleter, 0, 12, 1, 6)
        grid.addWidget(self.label_reference_viewer, 1, 0, 6, 6)
        grid.addWidget(self.label_STT_result_viewer, 1, 6, 6, 6)
        grid.addWidget(self.label_analysis_result_viewer, 1, 12, 6, 6)

    def set_reference_viewer(self):
        table_reference_viewer = QtWidgets.QTableWidget()
        return table_reference_viewer

    def set_STT_result_viewer(self):
        table_STT_result_viewer = QtWidgets.QTableWidget()
        return table_STT_result_viewer

    def set_analysis_result_viewer(self):
        table_analysis_result_viewer = QtWidgets.QTableWidget()
        return table_analysis_result_viewer

    def set_importer(self):
        importer = QtWidgets.QPushButton('import')
        importer.clicked.connect(self.FUNCTION)
        return importer

    def set_calculator(self):
        calculator = QtWidgets.QPushButton('calculate')
        calculator.clicked.connect(self.FUNCTION)
        return calculator

    def set_deleter(self):
        deleter = QtWidgets.QPushButton('delete')
        deleter.clicked.connect(self.FUNCTION)
        return deleter

    def FUNCTION(self):
        pass

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = STTApp()
    window.show()
    app.exec_()