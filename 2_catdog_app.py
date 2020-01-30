import sys

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

        self.string_file_name = None
        self.data_image = None
        self.string_class_name = None
        self.label_dict = ['Cat', 'Dog']

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
        image_classifier.clicked.connect(self.classify_image)
        return image_classifier

    def load_image(self):
        self.string_file_name = QtWidgets.QFileDialog.\
            getOpenFileName(self, 'Open File', './', self.tr('Images (*.png *.jpg *.gif)'))[0]
        self.label_image_viewer.setPixmap(QtGui.QPixmap(self.string_file_name))

    def classify_image(self):
        # load image data
        img = np.expand_dims(self.center_crop(self.resize(cv2.imread(self.string_file_name))), axis=0)
        # run inference session
        prediction = self.sess.run(self.prediction, feed_dict={self.X: img})
        self.string_class_name = self.label_dict[int(prediction)]
        self.label_class_viewer.setText('class : ' + self.string_class_name)

    def resize(self, img):
        h = img.shape[0]
        w = img.shape[1]
        if h > w:
            return cv2.resize(img, (256, int(h / w * 256)), interpolation=cv2.INTER_LINEAR)
        else:
            return cv2.resize(img, (int(w / h * 256), 256), interpolation=cv2.INTER_LINEAR)

    def center_crop(self, img):
        h = img.shape[0]
        w = img.shape[1]
        return img[h // 2 - 113:h // 2 + 114, w // 2 - 113:w // 2 + 114, :]


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = CatDogApp()
    window.show()
    app.exec_()