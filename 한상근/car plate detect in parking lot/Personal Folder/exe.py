from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QDesktopWidget, QHBoxLayout, QVBoxLayout, QLabel, QAction,
                            QFileDialog, QGridLayout, QScrollArea, QTextBrowser, QPushButton, QGroupBox, QLineEdit, QComboBox, QCalendarWidget, QRadioButton)
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSignal, Qt, pyqtSlot, QThread, QDate

from car_plate_detect import yolo_start
import multiprocessing as mp
import os, sys, time, sqlite3


class date_select(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.show()

    def initUI(self):
        self.setWindowTitle('날짜 선택')
        self.setGeometry(300,300, 500, 300)
        self.setFixedSize(500, 300)

        self.in_date_button = QRadioButton('입차시간 기준 검색', self)
        self.out_date_button = QRadioButton('출차시간 기준 검색', self)
        self.in_date_button.setChecked(True)

        hbox_button = QHBoxLayout()
        hbox_button.addWidget(self.in_date_button)
        hbox_button.addWidget(self.out_date_button)


        self.start_label = QLabel('검색 시작 날짜', self)
        self.start_label.setAlignment(Qt.AlignCenter)
        self.end_label = QLabel('검색 종료 날짜', self)
        self.end_label.setAlignment(Qt.AlignCenter)

        hbox_label = QHBoxLayout()
        hbox_label.addWidget(self.start_label)
        hbox_label.addWidget(self.end_label)

        self.start_day = QCalendarWidget()
        self.end_day = QCalendarWidget()
        self.start_day.setGridVisible(True)
        self.end_day.setGridVisible(True)
        self.start_day.setVerticalHeaderFormat(0)
        self.end_day.setVerticalHeaderFormat(0)

        self.start_day.clicked.connect(self.calendar_choose_start)
        self.end_day.clicked.connect(self.calendar_choose_end)

        hbox_date = QHBoxLayout()
        hbox_date.addWidget(self.start_day)
        hbox_date.addStretch(1)
        hbox_date.addWidget(self.end_day)

        quit_button = QPushButton('확인', self)
        quit_button.clicked.connect(self.emit_date)
        self.quit_button_alarm = QLabel('검색 시작 날짜와 종료 날짜를 선택해 주세요', self)
        self.quit_button_alarm.setAlignment(Qt.AlignCenter)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_button)
        vbox.addLayout(hbox_label)
        vbox.addLayout(hbox_date)
        vbox.addWidget(self.quit_button_alarm)
        vbox.addWidget(quit_button)

        self.choosen_time_sec_start = '0000-00-00'
        self.choosen_time_sec_end = '0000-00-00'

        self.setLayout(vbox)

    def calendar_choose_start(self):
        self.choosen_time_sec_start = self.start_day.selectedDate().toPyDate()
        parking_lot_app.log_search_window.search_label.setText(str(self.choosen_time_sec_start) + ' ~ ' + str(self.choosen_time_sec_end))
        self.time_range_start = time.mktime(self.choosen_time_sec_start.timetuple())

    def calendar_choose_end(self):
        self.choosen_time_sec_end = self.end_day.selectedDate().toPyDate()
        parking_lot_app.log_search_window.search_label.setText(str(self.choosen_time_sec_start) + ' ~ ' + str(self.choosen_time_sec_end))
        self.time_range_end = time.mktime(self.choosen_time_sec_end.timetuple()) + 60*60*24

    def emit_date(self):
        if self.choosen_time_sec_start == '0000-00-00':
            self.quit_button_alarm.setText('검색 시작 날짜를 선택해주세요.')
        elif self.choosen_time_sec_end == '0000-00-00':
            self.quit_button_alarm.setText('검색 종료 날짜를 선택해주세요.')
        else:
            if self.in_date_button.isChecked():
                parking_lot_app.log_search_window.take_date(self.time_range_start, self.time_range_end, 'time_in_sec')
                self.close()
            else:
                parking_lot_app.log_search_window.take_date(self.time_range_start, self.time_range_end, 'time_out_sec')
                self.close()

class show_image(QWidget):
    def __init__(self, file_path):
        super().__init__()
        self.setWindowTitle('차량 이미지')
        self.setGeometry(300,300, 500,500)

        self.image_pix = QPixmap(file_path)
        self.image_pix = self.image_pix.scaled(400, 400, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        self.image = QLabel()
        self.image.setPixmap(self.image_pix)
        hbox = QHBoxLayout()
        hbox.addWidget(self.image)
        self.setLayout(hbox)
        self.show()

class log_search_window(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.show()

    def initUI(self):
        self.setWindowTitle('주차장 입출입 로그')
        self.resize(1000,800)
        self.center()

        self.search_label = QLabel('검색 대기중', self)
        self.search_subject = QComboBox(self)
        self.search_subject.addItem('검색 옵션을 선택해주세요')
        self.search_subject.addItem('주차장 내 차량만 확인하기')
        self.search_subject.addItem('차량 번호판 검색')
        self.search_subject.addItem('날짜 검색')

        self.search_subject.activated[str].connect(self.search_options)

        self.search_engine = QLineEdit()
        self.search_engine.setFixedWidth(100)
        self.search_engine.returnPressed.connect(self.change_search_subject)

        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setWidgetResizable(True)

        self.scroll_box = QVBoxLayout()
        self.wid = QWidget()

        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.search_subject)
        self.hbox.addWidget(self.search_engine)
        self.hbox.addWidget(self.search_label)

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.hbox)
        self.vbox.addWidget(self.scroll)
        
        self.setLayout(self.vbox)

    def change_search_subject(self):
        if '차량 번호판 검색' == self.search_subject.currentText():
            self.scroll_box.deleteLater()
            self.scroll_box = QVBoxLayout()
            self.wid.deleteLater()
            self.wid = QWidget()
            self.search_label.setText('검색: ' + self.search_engine.text())
            parking_db = sqlite3.connect('parking_lot.db')
            db_io = parking_db.cursor()
            data_to_insert = ["%" + self.search_engine.text() + "%"]
            db_io.execute("SELECT id, car_plate, time_in, time_out, price, detected_image_path_in_car, detected_image_path_out_car FROM car_plate_data WHERE car_plate LIKE ?", data_to_insert)
            logs = db_io.fetchall()
            for log in logs:
                self.update_logs_by_plate(log)
            parking_db.close()
            self.search_label.setText('주차장 내 차량 검색 완료')
            self.wid.setLayout(self.scroll_box)
            self.scroll.setWidget(self.wid)

    def update_logs_by_plate(self, log):
        #car_log_num = QLabel("로그번호: {}".format(log[0]))
        car_plate = QLabel("번호판: {}".format(log[1]))
        car_in_time = QLabel("입차시간: {}".format(log[2]))
        car_out_time = QLabel("출차시간: {}".format(log[3]))
        car_price = QLabel("요금: {}".format(log[4]))
        file_path_in = QPushButton('입차 이미지 확인', self)
        file_path_out = QPushButton('출차 이미지 확인', self)
        file_path_in.clicked.connect(lambda: self.open_image(log[5]))
        file_path_out.clicked.connect(lambda: self.open_image(log[6]))
        
        hbox = QHBoxLayout()
        #hbox.addWidget(car_log_num)
        hbox.addWidget(car_plate)
        hbox.addWidget(car_in_time)
        hbox.addWidget(car_out_time)
        hbox.addWidget(car_price)
        hbox.addWidget(file_path_in)
        hbox.addWidget(file_path_out)
        self.scroll_box.addLayout(hbox)
        self.wid.setLayout(self.scroll_box)
        self.scroll.setWidget(self.wid)

    def search_options(self, option):
        self.scroll_box.deleteLater()
        self.scroll_box = QVBoxLayout()
        self.wid.deleteLater()
        self.wid = QWidget()
        self.search_label.setText('검색 대기중')
        if option == '주차장 내 차량만 확인하기':
            parking_db = sqlite3.connect('parking_lot.db')
            db_io = parking_db.cursor()
            db_io.execute("SELECT id, car_plate, time_in, price, detected_image_path_in_car FROM car_plate_data WHERE time_out = ' '")
            logs = db_io.fetchall()
            for log in logs:
                self.update_logs_in_parking_lot(log)
            parking_db.close()
            self.search_label.setText('주차장 내 차량 검색 완료')

        elif option == '날짜 검색':
            self.date_select = date_select()

    def take_date(self, time_start, time_end, car_in_out):
        self.scroll_box.deleteLater()
        self.scroll_box = QVBoxLayout()
        self.wid.deleteLater()
        self.wid = QWidget()

        parking_db = sqlite3.connect('parking_lot.db')
        db_io = parking_db.cursor()
        if car_in_out == 'time_in_sec':
            db_io.execute("SELECT id, car_plate, time_in, time_out, price, detected_image_path_in_car, detected_image_path_out_car FROM car_plate_data WHERE time_in_sec >= ? AND time_in_sec <= ?", [time_start, time_end])
        else:
            db_io.execute("SELECT id, car_plate, time_in, time_out, price, detected_image_path_in_car, detected_image_path_out_car FROM car_plate_data WHERE time_out_sec >= ? AND time_out_sec <= ?", [time_start, time_end])
        
        logs = db_io.fetchall()
        for log in logs:
            self.update_logs_in_parking_lot(log)
        parking_db.close()

            
    
    def update_logs_in_parking_lot(self, log):
        #car_log_num = QLabel("로그번호: {}".format(log[0]))
        car_plate = QLabel("번호판: {}".format(log[1]))
        car_in_time = QLabel("입차시간: {}".format(log[2]))
        car_price = QLabel("요금: {}".format(log[3]))
        file_path = QPushButton('이미지 확인', self)
        file_path.clicked.connect(lambda: self.open_image(log[4]))
        
        hbox = QHBoxLayout()
        #hbox.addWidget(car_log_num)
        hbox.addWidget(car_plate)
        hbox.addWidget(car_in_time)
        hbox.addWidget(car_price)
        hbox.addWidget(file_path)
        self.scroll_box.addLayout(hbox)
        self.wid.setLayout(self.scroll_box)
        self.scroll.setWidget(self.wid)


    def open_image(self, file_path):
        self.image_widget_open = show_image(file_path)

    def center(self):
        current_location = self.frameGeometry()
        location_center = QDesktopWidget().availableGeometry().center()
        current_location.moveCenter(location_center)
        self.move(current_location.topLeft())

    
class parking_lot(QMainWindow):
    def __init__(self, exit_log_class, enter_log_class):
        super().__init__()
        self.initUI()

        detail = details(exit_log_class, enter_log_class)
        self.setCentralWidget(detail)
        self.show()

    def initUI(self):
        self.setWindowTitle('자동차 번호판 탐지')
        self.resize(800,600)
        self.center()

        open_video = QAction(QIcon('open.png'), '파일 열기', self)
        open_video.triggered.connect(self.start_yolo_detection)

        open_logs = QAction('전체 로그 확인', self)
        open_logs.triggered.connect(self.open_logs_window)

        menu = self.menuBar()
        menu.setNativeMenuBar(False)
        filemenu = menu.addMenu('파일')
        filemenu.addAction(open_video)
        filemenu.addAction(open_logs)


    def start_yolo_detection(self):
        filename = QFileDialog.getOpenFileName(self, '비디오 파일 선택', './')

        yolo_process = mp.Process(name='yolo', target=yolo_start, args=(filename[0], ), daemon=True)
        yolo_process.start()

    def open_logs_window(self):
        self.log_search_window = log_search_window()

    def center(self):
        current_location = self.frameGeometry()
        location_center = QDesktopWidget().availableGeometry().center()
        current_location.moveCenter(location_center)
        self.move(current_location.topLeft())

def enter_log_producer(enter_log_class):
    id = 1
    while True:
        parking_db = sqlite3.connect('parking_lot.db')
        db_io = parking_db.cursor()

        id_search = [id]
        db_io.execute("SELECT car_plate, detected_image_path_in_car, time_in FROM car_plate_data WHERE id = ?", id_search)
        log = db_io.fetchone()
        if log:
            enter_log_class.put(log)
            id += 1
        else:
            parking_db.close()
            time.sleep(1)


def exit_log_producer(exit_log_class):

    parking_db = sqlite3.connect('parking_lot.db')
    db_io = parking_db.cursor()

    db_io.execute("SELECT car_plate, detected_image_path_out_car, time_out FROM car_plate_data WHERE time_out_sec != 0 ORDER BY time_out_sec ASC")
    logs = db_io.fetchall()
    temp = 0
    if logs:
        for log in logs:
            exit_log_class.put(log)
        temp = log

    while True:
        parking_db = sqlite3.connect('parking_lot.db')
        db_io = parking_db.cursor()
        db_io.execute("SELECT car_plate, detected_image_path_out_car, time_out FROM car_plate_data WHERE time_out_sec != 0 AND time_out_sec = (SELECT max(time_out_sec) FROM car_plate_data)")
        log = db_io.fetchone()
        if log:
            if log == temp:
                parking_db.close()
                time.sleep(1)
            else:
                temp = log
                exit_log_class.put(log)


class enter_log_consumer(QThread):
    poped = pyqtSignal(tuple)

    def __init__(self, enter_log_class):
        super().__init__()
        self.enter_log_class = enter_log_class

    def run(self):
        while True:
            if not self.enter_log_class.empty():
                data = self.enter_log_class.get()
                self.poped.emit(data)

class exit_log_consumer(QThread):
    poped = pyqtSignal(tuple)

    def __init__(self, exit_log_class):
        super().__init__()
        self.exit_log_class = exit_log_class

    def run(self):
        while True:
            if not self.exit_log_class.empty():
                data = self.exit_log_class.get()
                self.poped.emit(data)



class details(QWidget):
    def __init__(self, exit_log_class, enter_log_class):
        super().__init__()
        self.initUI(exit_log_class, enter_log_class)
        self.show()

    def initUI(self, exit_log_class, enter_log_class):
        self.setWindowTitle('자동차 번호판 탐지')

        grid = QGridLayout()
        grid.addWidget(car_in_image_widget(enter_log_class), 0, 0)
        grid.addWidget(car_out_image_widget(exit_log_class), 0, 1)

        self.setLayout(grid)

class car_in_image_widget(QWidget):
    def __init__(self, enter_log_class):
        super().__init__()
        self.initUI()
        self.show()
        self.consumer = enter_log_consumer(enter_log_class)
        self.consumer.poped.connect(self.update_logs)
        self.consumer.start()

    def initUI(self):
        self.image_label = QLabel('들어오는 차량', self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_pix = QPixmap('test.jpg')
        self.image_pix = self.image_pix.scaled(400, 400, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        self.image = QLabel()
        self.image.setPixmap(self.image_pix)
        self.image.setAlignment(Qt.AlignCenter)

        self.subject_label = QLabel('들어온 차량 기록', self)
        self.subject_label.setAlignment(Qt.AlignCenter)
        

        self.scroll = QScrollArea()

        self.scroll.setMinimumHeight(400)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setWidgetResizable(True)

        self.scroll_box = QVBoxLayout()
        self.wid = QWidget()

        self.vbox = QVBoxLayout()
        self.vbox.addStretch(100)
        self.vbox.addWidget(self.image_label)
        self.vbox.addWidget(self.image)
        self.vbox.addStretch(100)
        self.vbox.addWidget(self.subject_label)
        self.vbox.addWidget(self.scroll)
        self.vbox.addStretch(100)

        self.setLayout(self.vbox)

    @pyqtSlot(tuple)
    def update_logs(self, log):
        car_plate = QLabel("{}".format(log[0]))
        car_time = QLabel("{}".format(log[2]))
        file_path = QPushButton('이미지 확인', self)
        file_path.clicked.connect(lambda: self.open_image(log[1]))

        hbox = QHBoxLayout()
        hbox.addWidget(car_plate)
        hbox.addWidget(car_time)
        hbox.addWidget(file_path)
        self.scroll_box.addLayout(hbox)
        self.wid.setLayout(self.scroll_box)
        self.scroll.setWidget(self.wid)

    def open_image(self, file_path):
        self.image_pix = QPixmap(file_path)
        self.image_pix = self.image_pix.scaled(400, 400, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        self.image.setPixmap(self.image_pix)
        self.image.setAlignment(Qt.AlignCenter)
    

class car_out_image_widget(QWidget):
    def __init__(self, exit_log_class):
        super().__init__()
        self.initUI()
        self.show()
        self.consumer = exit_log_consumer(exit_log_class)
        self.consumer.poped.connect(self.update_logs)
        self.consumer.start()

    def initUI(self):
        self.image_label = QLabel('나가는 차량', self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_pix = QPixmap('test.jpg')
        self.image_pix = self.image_pix.scaled(400, 400, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        self.image = QLabel()
        self.image.setPixmap(self.image_pix)
        self.image.setAlignment(Qt.AlignCenter)

        self.subject_label = QLabel('나간 차량 기록', self)
        self.subject_label.setAlignment(Qt.AlignCenter)
        

        self.scroll = QScrollArea()
        
        #self.scroll.setFixedHeight(500)
        self.scroll.setMinimumHeight(400)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setWidgetResizable(True)

        self.scroll_box = QVBoxLayout()
        self.wid = QWidget()

        self.vbox = QVBoxLayout()
        self.vbox.addStretch(100)
        self.vbox.addWidget(self.image_label)
        self.vbox.addWidget(self.image)
        self.vbox.addStretch(100)
        self.vbox.addWidget(self.subject_label)
        self.vbox.addWidget(self.scroll)
        self.vbox.addStretch(100)

        self.setLayout(self.vbox)

    @pyqtSlot(tuple)
    def update_logs(self, log):
        car_plate = QLabel("{}".format(log[0]))
        car_time = QLabel("{}".format(log[2]))
        file_path = QPushButton('이미지 확인', self)
        file_path.clicked.connect(lambda: self.open_image(log[1]))

        hbox = QHBoxLayout()
        hbox.addWidget(car_plate)
        hbox.addWidget(car_time)
        hbox.addWidget(file_path)
        self.scroll_box.addLayout(hbox)
        self.wid.setLayout(self.scroll_box)
        self.scroll.setWidget(self.wid)

    def open_image(self, file_path):
        self.image_pix = QPixmap(file_path)
        self.image_pix = self.image_pix.scaled(400, 400, Qt.IgnoreAspectRatio, Qt.FastTransformation)
        self.image.setPixmap(self.image_pix)
        self.image.setAlignment(Qt.AlignCenter)



if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    mp.freeze_support()
    mp.set_start_method('spawn')

    if not os.path.exists('parking_lot.db'):
        parking_db = sqlite3.connect('parking_lot.db')
        parking_db.execute('CREATE TABLE car_plate_data(id INTEGER, car_plate TEXT, detected_image_path_in_plate TEXT, time_in TEXT, detected_image_path_out_plate TEXT, time_out TEXT, time_in_sec REAL, time_out_sec REAL, price INTEGER, detected_image_path_in_car TEXT, detected_image_path_out_car TEXT)')
        db_io = parking_db.cursor()
        #db_io.execute("INSERT INTO car_plate_data VALUES (0, 'TEST', 'TEST', 'TEST', 'TEST', 'TEST', 0, 0, 0, 'TEST', 'TEST')")
        parking_db.close()
        time.sleep(2)
    

    enter_log_class = mp.Queue()
    exit_log_class = mp.Queue()

    enter_log_process = mp.Process(name='enter_log_producer', target=enter_log_producer, args=(enter_log_class,), daemon=True)
    exit_log_process = mp.Process(name='exit_log_producer', target=exit_log_producer, args=(exit_log_class,), daemon=True)
    enter_log_process.start()
    exit_log_process.start()

    app = QApplication(sys.argv)
    parking_lot_app = parking_lot(exit_log_class, enter_log_class)
    parking_lot_app.show()
    sys.exit(app.exec_())