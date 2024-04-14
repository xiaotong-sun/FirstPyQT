import os
import sys
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFileDialog,
    QMainWindow,
)
from PyQt6.QtGui import QPixmap, QIcon, QFont
from PyQt6.QtCore import (
    Qt,
    QThread,
    QEventLoop,
    QTimer,
    QSize,
    QCoreApplication,
    QEvent,
)
from qfluentwidgets import *
from PIL import Image

import time
from main import searchForImg


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.open_file_name = ""
        self.initUI()

    def initUI(self):
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.resize(1280, 720)
        self.center()
        self.setWindowIcon(QIcon("./icon/icon.png"))

        self.openScreen = SplashScreen(self.windowIcon(), self, False)
        self.openScreen.setIconSize(QSize(1020, 1020))
        self.show()
        self.createSubInterface()
        self.openScreen.finish()
        self.showMaximized()

        self.select_button = PushButton(FluentIcon.IMAGE_EXPORT, " 选择图片")
        self.select_button.setIconSize(QSize(22, 22))
        self.process_button = PushButton(FluentIcon.SEND, " 处理数据")
        self.process_button.setIconSize(QSize(22, 22))
        self.clear_button = PushButton(FluentIcon.BROOM, " 清除图片")
        self.clear_button.setIconSize(QSize(22, 22))
        self.exit_button = PushButton(FluentIcon.POWER_BUTTON, " 退出系统")
        self.exit_button.setIconSize(QSize(22, 22))
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sidebar_image_label = ClickableLabel()
        self.sidebar_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sidebar_image_label.installEventFilter(self.sidebar_image_label)

        # 修改按钮的大小及字体
        font = QFont()
        font.setPointSize(15)
        self.select_button.setFixedSize(200, 100)
        self.process_button.setFixedSize(200, 100)
        self.clear_button.setFixedSize(200, 100)
        self.exit_button.setFixedSize(200, 100)
        self.select_button.setFont(font)
        self.process_button.setFont(font)
        self.clear_button.setFont(font)
        self.exit_button.setFont(font)

        # 添加 widgets 到布局
        self.left_layout = QVBoxLayout()  # 使用 QVBoxLayout 来垂直排列 widgets
        self.left_layout.addWidget(self.select_button)
        self.left_layout.addWidget(self.process_button)
        self.left_layout.addWidget(self.clear_button)
        self.left_layout.addWidget(self.exit_button)
        self.left_layout.addWidget(self.sidebar_image_label)

        self.right_layout = QVBoxLayout()
        self.right_layout.addWidget(self.image_label)

        # 调整布局
        self.left_layout.addStretch(5)
        self.left_layout.setSpacing(60)
        self.left_widget = QWidget()
        self.left_widget.setLayout(self.left_layout)
        self.left_widget.setStyleSheet("QWidget { background-color: #A8DADC ; border-radius: 15px}")

        self.right_widget = QWidget()
        self.right_widget.setLayout(self.right_layout)
        self.right_widget.setStyleSheet(
            "QWidget { background-color: #F1FAEE ; border-radius: 15px}"
        )

        # 设置主布局
        self.main_layout = QHBoxLayout()
        self.main_layout.addWidget(self.left_widget)
        self.main_layout.addWidget(self.right_widget)
        self.main_layout.setStretch(0, 10)
        self.main_layout.setStretch(1, 200)
        self.main_layout.setSpacing(15)
        self.main_weiget = QWidget()
        self.main_weiget.setLayout(self.main_layout)
        self.main_weiget.setStyleSheet("QWidget { background-color: #45A29E ;}")
        self.setCentralWidget(self.main_weiget)

        # 绑定按钮点击事件
        self.select_button.clicked.connect(self.open_image_file)
        self.process_button.clicked.connect(self.process_data)
        self.clear_button.clicked.connect(self.clear_image)
        self.exit_button.clicked.connect(self.close)

    def createSubInterface(self):
        loop = QEventLoop(self)
        QTimer.singleShot(800, loop.quit)
        loop.exec()

    def center(self):
        '''设置窗口启动于中间位置'''
        qr = self.frameGeometry()
        cp = QCoreApplication.instance().primaryScreen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def open_image_file(self):
        self.clear_image()
        # 打开文件选择对话框
        self.open_file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.gif)"
        )
        print(self.open_file_name)
        if self.open_file_name:
            # 当选择了文件后，创建一个 QPixmap 对象并将其设置到 QLabel 中显示
            pixmap = QPixmap(self.open_file_name)
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.width(),
                    self.image_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            )

    def process_data(self):
        if self.open_file_name == "":
            error_message = MessageBox("Error", "请先选择要查询的图片!", self)
            error_message.cancelButton.hide()
            error_message.exec()
        else:
            img = os.path.basename(self.open_file_name)
            print(img)
            utm = self.getUtm(img)
            print(utm)
            if utm != "":
                self.progress_dialog = CustomMessageBox(self)
                self.data_processor = DataProcessor(img=img, utm=utm)
                self.data_processor.finished.connect(self.close_progress_dialog)
                self.data_processor.start()
                self.progress_dialog.move(
                    self.frameGeometry().center() - self.progress_dialog.rect().center()
                )
                self.progress_dialog.exec()

                # 处理完成之后显示图片
                self.sidebar_image_label.open_image = self.open_file_name
                self.show_image()
            else:
                error_message = MessageBox(
                    "Error", "选择的图片无效，请选择data/query文件夹中的图片", self
                )
                error_message.cancelButton.hide()
                error_message.exec()

    def close_progress_dialog(self):
        self.progress_dialog.change()

    def clear_image(self):
        self.image_label.clear()
        self.sidebar_image_label.clear()
        self.open_file_name = ""
        self.sidebar_image_label.open_image = ""

    def show_image(self):
        self.image_label.clear()
        self.sidebar_image_label.clear()
        pixmap = QPixmap('C:/Users/84097/Desktop/1.jpg')
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
        )
        pixmap = QPixmap(self.open_file_name)
        self.sidebar_image_label.setPixmap(
            pixmap.scaled(
                190,
                190,
                Qt.AspectRatioMode.KeepAspectRatio,
            )
        )

    def getUtm(self, img):
        loaded_dict = {}
        with open('./data/query_dict.txt', 'r') as file:
            for line in file:
                key, value = line.strip().split(': ')
                loaded_dict[key] = value

        try:
            utm = loaded_dict[img]
        except KeyError as e:
            return ""
        else:
            return utm


class ClickableLabel(QLabel):
    '''设置sidebar_image_label可以点击展开, 以方便查看'''

    open_image = ""

    def __init__(self):
        super().__init__()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseButtonPress:
            if obj == self:
                if self.open_image == "":
                    pass
                else:
                    self.showFlyout(obj)
            return True
        return super().eventFilter(obj, event)

    def showFlyout(self, obj):
        view = FlyoutView(
            image=self.open_image,
            title="",
            content="",
            isClosable=False,
        )
        view.widgetLayout.insertSpacing(1, 5)
        view.widgetLayout.addSpacing(5)
        button = PushButton('关闭')
        button.setFixedWidth(120)
        view.addWidget(button, align=Qt.AlignmentFlag.AlignBottom)
        view.setFixedWidth(600)
        view.setMaximumHeight(500)
        view.imageLabel.scaledToWidth(600)

        w = Flyout.make(view, obj, self, aniType=FlyoutAnimationType.SLIDE_RIGHT)
        view.closed.connect(w.close)
        button.clicked.connect(w.close)


class DataProcessor(QThread):
    '''数据处理部分，用于将选择的图片传入，模型处理，并输出处理结果'''

    def __init__(self, img, utm):
        super().__init__()
        self.img = img
        self.utm = utm

    def run(self):
        # 调用模型的test，返回得到的文件路径
        image_paths = searchForImg(self.img, self.utm)
        # folder_path = 'C:/Users/84097/Desktop/figure/'
        # image_paths = [
        #     os.path.join(folder_path, f)
        #     for f in os.listdir(folder_path)
        #     if os.path.isfile(os.path.join(folder_path, f))
        # ]

        combined_image = self.combine_images(image_paths)
        combined_image.save('./combined_image.jpg')

        self.finished.emit()

    def combine_images(self, image_paths):
        images = [Image.open(image_path) for image_path in image_paths]

        # 调整图片大小以适应最大图片的尺寸
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)

        resized_images = [img.resize((max_width, max_height)) for img in images]

        # 创建新的合并图片
        rows, cols = 4, 5
        combined_width = max_width * cols
        combined_height = max_height * rows
        combined_image = Image.new('RGB', (combined_width, combined_height))

        # 将调整大小后的图片按照行列顺序合并
        for i in range(len(resized_images)):
            col = i % cols
            row = i // cols
            x_offset = col * max_width
            y_offset = row * max_height
            combined_image.paste(resized_images[i], (x_offset, y_offset))

        return combined_image


class CustomMessageBox(MessageBoxBase):
    '''自定义消息弹窗'''

    def __init__(self, parent=None):
        super().__init__(parent)
        self.titleLabel = SubtitleLabel('正在处理中，请耐心等待...', self)
        self.bar = IndeterminateProgressBar(start=True)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.bar)

        self.widget.setMinimumWidth(350)
        self.yesButton.setText('等待')
        self.yesButton.setDisabled(True)
        self.cancelButton.hide()

    def change(self):
        self.titleLabel.setText('处理完成')
        self.progressBar = ProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(100)
        self.progressBar.setCustomBarColor(
            QColor(0, 175, 110),
            QColor(0, 175, 110),
        )
        self.viewLayout.replaceWidget(self.bar, self.progressBar)
        self.yesButton.setText('确认')
        self.yesButton.setDisabled(False)


if __name__ == "__main__":
    os.environ['QT_IMAGEIO_MAXALLOC'] = str(10**9)
    app = QApplication(sys.argv)
    viewer = MainWindow()
    viewer.show()
    sys.exit(app.exec())
