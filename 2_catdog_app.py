import sys
import time

import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui

import tensorflow as tf
import numpy as np
import cv2

import model

class CatDogApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        ### image IO parameters
        self.string_file_name = None
        self.data_image = None
        self.string_class_name = None
        self.label_dict = ['Cat', 'Dog']

        ### dark parameters
        self.dark_flag = False
        self.dark_reference_image = cv2.imread('dark.JPG')
        self.dark_transform_matrix = np.identity(3)

        ### neural network parameters
        self.model = model.AlexNetModel(input_size=1)
        self.sess = self.run_session()

        self.initUI()

    def run_session(self):
        self.X = tf.placeholder(shape=[None, 227, 227, 3], dtype=tf.float32)
        self.prediction = self.make_inference_graph(self.X)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        return sess

    def make_inference_graph(self, input_image):
        logit = self.model.classifier(input_image, keep_prob=1.0)
        return tf.argmax(logit, axis=1)

    def initUI(self):
        # set window title
        self.setWindowTitle('Cat Dog App')

        # set title bar icon
        self.setWindowIcon(QtGui.QIcon('dark.JPG'))

        # set geometry
        self.setGeometry(100, 100, 300, 500)

        # set image viewer label
        self.label_image_viewer = self.set_image_viewer()

        # set class viewer label
        self.label_class_viewer = self.set_class_viwer()

        # set image loader button
        self.btn_image_loader = self.set_image_loader()

        # set image classifier button
        self.btn_image_classifier = self.set_image_classifier()

        # set image resizer input line
        self.line_image_resizer = QtWidgets.QLineEdit()

        # set image resizer button
        self.btn_image_resizer = self.set_image_resizer()

        # set image rotater input line
        self.line_image_rotater = QtWidgets.QLineEdit()

        # set image rotater button
        self.btn_image_rotater = self.set_image_rotater()

        # set image v fliper button
        self.btn_image_vfliper = self.set_image_vfliper()

        # set image h fliper button
        self.btn_image_hfliper = self.set_image_hfliper()

        # set DARK button
        self.btn_darker = self.set_darker()

        # set layout
        grid = QtWidgets.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.label_image_viewer, 0, 0, 4, 4)
        grid.addWidget(self.label_class_viewer, 5, 0, 1, 4)
        grid.addWidget(self.btn_image_loader, 6, 0, 1, 3)
        grid.addWidget(self.btn_image_classifier, 6, 4, 1, 3)
        grid.addWidget(self.btn_image_resizer, 0, 5)
        grid.addWidget(self.line_image_resizer, 0, 6)
        grid.addWidget(self.btn_image_rotater, 1, 5)
        grid.addWidget(self.line_image_rotater, 1, 6)
        grid.addWidget(self.btn_image_vfliper, 2, 5, 1, 2)
        grid.addWidget(self.btn_image_hfliper, 3, 5, 1, 2)
        grid.addWidget(self.btn_darker, 4, 5, 1, 2)

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
        image_classifier.clicked.connect(self.classify_image)
        return image_classifier

    def set_image_resizer(self):
        image_resizer = QtWidgets.QPushButton('RESIZE(%)')
        image_resizer.clicked.connect(self.resize_image)
        return image_resizer

    def set_image_rotater(self):
        image_rotater = QtWidgets.QPushButton('ROTATE(°)')
        image_rotater.clicked.connect(self.rotate_image)
        return image_rotater

    def set_image_vfliper(self):
        image_vfliper = QtWidgets.QPushButton('VERTICAL FLIP')
        image_vfliper.clicked.connect(self.vflip_image)
        return image_vfliper

    def set_image_hfliper(self):
        image_hfliper = QtWidgets.QPushButton('HORIZONTAL FLIP')
        image_hfliper.clicked.connect(self.hflip_image)
        return image_hfliper

    def set_darker(self):
        darker = QtWidgets.QPushButton('')
        darker.setIcon(QtGui.QIcon('dark.JPG'))
        darker.setIconSize(QtCore.QSize(24, 24))
        darker.clicked.connect(self.dark)
        return darker

    def dark(self):
        if self.data_image is None or not self.dark_flag:
            self.data_image = self.dark_reference_image
            self.dark_flag = True

        timeLine = QtCore.QTimeLine(10000, self.label_image_viewer)
        timeLine.setFrameRange(0, 10000)
        timeLine.frameChanged.connect(self.animate_label)
        timeLine.start()

    def animate_label(self):
        h = self.data_image.shape[0]
        w = self.data_image.shape[1]
        matrix_rotate = cv2.getRotationMatrix2D((w // 2, h // 2), 15, 1.0)
        matrix_scale = np.array([[1.01, 0], [0, 1.01]])
        self.update_transform_matrix(np.dot(matrix_scale, matrix_rotate))
        self.data_image = cv2.warpAffine(self.dark_reference_image,
                                         self.dark_transform_matrix[:-1, :],
                                         (int(w * 1.01), int(h * 1.01)))
        self.display_image()

    def update_transform_matrix(self, mat):
        mat = np.concatenate((mat, np.expand_dims([0, 0, 1], axis=0)), axis=0)
        self.dark_transform_matrix = np.dot(mat, self.dark_transform_matrix)

    def load_image(self):
        file_name = QtWidgets.QFileDialog. \
            getOpenFileName(self, 'Open File', './', self.tr('Images (*.png *.jpg *.gif)'))[0]
        if file_name is '':
            return
        else:
            self.string_file_name = file_name
        self.data_image = cv2.imread(self.string_file_name)
        self.dark_flag = False
        self.display_image()

    def display_image(self):
        if self.data_image is None:
            QtWidgets.QMessageBox.warning(self, "WARNING", "There is no image")
            return

        img = QtGui.QImage(self.data_image, self.data_image.shape[1], self.data_image.shape[0],
                           self.data_image.shape[1] * 3, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.label_image_viewer.setPixmap(QtGui.QPixmap(img))

    def classify_image(self):
        # load image data
        img = np.expand_dims(self.center_crop(self.resize(self.data_image)), axis=0)
        # run inference session
        prediction = self.sess.run(self.prediction, feed_dict={self.X: img})
        self.string_class_name = self.label_dict[int(prediction)]
        self.label_class_viewer.setText('Class : ' + self.string_class_name)

    def resize_image(self):
        scale = self.line_image_resizer.text()
        if scale.isdecimal():
            scale = eval(scale)
        else:
            QtWidgets.QMessageBox.warning(self, "WARNING", "Resize-Input is not digit")
            return

        h = self.data_image.shape[0]
        w = self.data_image.shape[1]

        self.data_image = cv2.resize(self.data_image, (w * scale // 100, h * scale // 100), interpolation=cv2.INTER_LINEAR)
        self.display_image()

    def rotate_image(self):
        degree = self.line_image_rotater.text()
        if degree.isdecimal():
            degree = eval(degree)
        else:
            QtWidgets.QMessageBox.warning(self, "WARNING", "Rotate-Input is not digit")
            return

        h = self.data_image.shape[0]
        w = self.data_image.shape[1]

        matrix_rotate = cv2.getRotationMatrix2D((w // 2, h // 2), degree, 1.0)
        self.data_image = cv2.warpAffine(self.data_image, matrix_rotate, (w, h))
        self.display_image()


    def vflip_image(self):
        self.data_image = cv2.flip(self.data_image, flipCode=0)
        self.display_image()

    def hflip_image(self):
        self.data_image = cv2.flip(self.data_image, flipCode=1)
        self.display_image()

    def resize(self, img, scale=256):
        h = img.shape[0]
        w = img.shape[1]
        if h > w:
            return cv2.resize(img, (scale, int(h / w * scale)), interpolation=cv2.INTER_LINEAR)
        else:
            return cv2.resize(img, (int(w / h * scale), scale), interpolation=cv2.INTER_LINEAR)

    def center_crop(self, img):
        h = img.shape[0]
        w = img.shape[1]
        return img[h // 2 - 113:h // 2 + 114, w // 2 - 113:w // 2 + 114, :]


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = CatDogApp()
    window.show()
    app.exec_()