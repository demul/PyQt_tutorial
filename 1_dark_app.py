import sys
from PyQt5.QtWidgets import QApplication, QMainWindow,\
    QPushButton, QToolTip, QMenuBar, QLabel, QAction, QDesktopWidget,\
    QFileDialog
from PyQt5.QtCore import QCoreApplication, QSize
from PyQt5.QtGui import QIcon, QFont, QPixmap

class DarkApp(QMainWindow):
    def __init__(self, pos_parent=None):
        super().__init__()
        self.initUI(pos_parent)

    def initUI(self, pos):
        # make sub window's list
        self.sub_windows = []

        # set window title
        self.setWindowTitle('Dark App')

        # set title bar icon
        icon = QIcon('dark.JPG')
        self.setWindowIcon(QIcon(icon))

        # set label
        ### if u want to show(pop-up) image,
        ### it's ok to make pixmap after self.show()
        ### but u must bind label with window before self.show()
        self.label = QLabel(self)
        self.label.setGeometry(100, 100, 400, 400)

        # make actions
        self.make_actions()

        # make status bar
        self.status_bar = self.statusBar()

        # make tool bar
        self.tool_bar = self.addToolBar('Exit')

        # set tool bar
        self.set_tool_bar()

        # make memu bar
        self.menu_bar = self.menuBar()

        # set menu bar
        self.set_menu_bar()

        # set geometric options
        # self.move(300, 400)
        # self.resize(400, 200)
        if pos is None:
            self.setGeometry(100, 100, 600, 600)
        else:
            self.setGeometry(pos.x() + 400, pos.y(), 600, 600)

        # set tooltip's font
        QToolTip.setFont(QFont('SansSerif', 20))

        # make quit button
        # self.btn = self.make_quit_button()

        # make fake quit button
        self.btn = self.make_fake_quit_button()

    def set_tool_bar(self):
        self.tool_bar.addAction(self.exit_action)

    def set_menu_bar(self):
        self.menu_bar.setNativeMenuBar(False)
        file_menu = self.menu_bar.addMenu('&File')
        file_menu.addAction(self.exit_action)

    def make_actions(self):
        self.exit_action = QAction(QIcon('dark.JPG'), 'Exit', self)
        self.exit_action.setShortcut('Ctrl+Q')
        self.exit_action.setStatusTip('Exit')
        # exit_action.triggered.connect(QCoreApplication.instance().quit)
        self.exit_action.triggered.connect(self.pop_status_text)

    def make_quit_button(self):
        btn = QPushButton('Quit', self)
        # btn.move(50, 50)
        # btn.resize(btn.sizeHint())
        size_hint = btn.sizeHint()
        btn.setGeometry(600 - size_hint.width(), 600 - size_hint.height(), size_hint.width(), size_hint.height())
        btn.setToolTip('Deep <b>Dark</b> Fantasy')
        btn.clicked.connect(QCoreApplication.instance().quit)
        return btn

    def make_fake_quit_button(self):
        btn = QPushButton('Quit', self)
        # btn.move(50, 50)
        # btn.resize(btn.sizeHint())
        size_hint = btn.sizeHint()
        btn.setGeometry(600 - size_hint.width(), 600 - size_hint.height(), size_hint.width(), size_hint.height())
        btn.setToolTip('Deep<br><b>Dark</b><br>Fantasy')
        btn.clicked.connect(self.pop_status_text)
        return btn

    def pop_status_text(self):
        self.status_bar.showMessage('Deep Dark')
        self.show_dark_img()

    def show_dark_img(self):
        pix = QPixmap()
        pix.load('dark.JPG')
        pix = pix.scaled(400, 400)
        self.label.setPixmap(pix)
        self.make_new_dark_app()

    def make_new_dark_app(self):
        sub_window = DarkApp(self.pos())
        self.sub_windows.append(sub_window)
        sub_window.show()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DarkApp()
    ex.show()
    app.exec_()