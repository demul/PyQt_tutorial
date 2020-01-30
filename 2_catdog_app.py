import sys
import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui


class CatDogApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.string_file_name = None
        self.data_image = None
        self.string_class_name = None

        self.initUI()

    def initUI(self):
        # set window title
        self.setWindowTitle('Cat Dog App')

        # set title bar icon
        self.setWindowIcon(QtGui.QIcon('dark.JPG'))

        # set geometry
        self.setGeometry(100, 100, 600, 900)

        # set image viewer label
        self.label_image_viewer = self.set_image_viewer()

        # set class viewer label
        self.label_class_viewer = self.set_class_viwer()

        # set image loader button
        self.btn_image_loader = self.set_image_loader()

        # set image classifier button
        self.btn_image_classifier = self.set_image_classifier()

        # set layout
        grid = QtWidgets.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.label_image_viewer, 0, 0, 4, 4)
        grid.addWidget(self.label_class_viewer, 5, 0, 1, 4)
        grid.addWidget(self.btn_image_loader, 6, 0, 1, 2)
        grid.addWidget(self.btn_image_classifier, 6, 3, 1, 2)

    def set_image_viewer(self):
        label_image_viewer = QtWidgets.QLabel()
        return label_image_viewer

    def set_class_viwer(self):
        label_class_viewer = QtWidgets.QLabel()
        return label_class_viewer

    def set_image_loader(self):
        image_loader = QtWidgets.QPushButton('LOAD')
        image_loader.clicked.connect(self.load_image)
        return image_loader

    def set_image_classifier(self):
        image_classifier = QtWidgets.QPushButton('CLASSIFY')
        # image_classifier.clicked.connect(self.classify_image)
        return image_classifier

    def load_image(self):
        self.string_file_name = QtWidgets.QFileDialog.\
            getOpenFileName(self, 'Open File', './', self.tr('Images (*.png *.jpg *.gif)'))[0]
        self.label_image_viewer.setPixmap(QtGui.QPixmap(self.string_file_name))

    # def classify_image(self):
    #     self.string_class_name = CLASSIFIER
    #     self.label_class_viewer.setText('class : ' + self.string_class_name)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = CatDogApp()
    window.show()
    app.exec_()