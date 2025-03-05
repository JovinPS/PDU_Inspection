import os,sys,cv2,io,time,json
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,QSpacerItem,QSizePolicy,
                             QLabel, QFrame, QGridLayout,QPushButton,QFileDialog,QMessageBox,QProgressBar,
                             QDesktopWidget)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt,QThread,pyqtSignal
# import os
from pdulive import get_status
from ultralytics import YOLO
from PIL import Image
# import io
# import time

  
class ResultBox(QWidget):    
    def __init__(self, label_text, status,count_items, prediction_dict, parent=None):
        super(ResultBox, self).__init__(parent)
        self.label_text = label_text
        self.status = status
        style_string=""" border-radius: 10px;
                 
                 color: black;font-weight: bold;padding: 5px 10px;"""
        self.layout = QHBoxLayout()
        self.label = QLabel(self.label_text)
        if self.label_text == 'PARTS':
            style_string = "font-size: 11pt;text-align: center;font-weight: bold;"
            style_string=""" border-radius: 10px;
                 background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(255, 204, 102, 1), stop:1 rgba(255, 165, 0, 1));
                 color: #800000;font-weight: bold;padding: 5px 10px;"""
            self.label.setStyleSheet(style_string)
        self.label.setStyleSheet(style_string)
        self.color_frame = QFrame()
        self.status_label = QLabel(self.status)
        if self.status == 'STATUS':
            style_string = """ border-radius: 10px;
                 background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(255, 204, 102, 1), stop:1 rgba(255, 165, 0, 1));
                 color: #800000;font-weight: bold;padding: 5px 10px;"""
            self.status_label.setStyleSheet(style_string)
        else :
            color=self.get_color()
            style_string=f""" border-radius: 10px;
                 
                 color: black;font-weight: bold;padding: 5px 10px;background-color: {color}"""
            self.status_label.setStyleSheet(style_string)
        self.copy_button = QPushButton("Copy Path")
        self.copy_button.clicked.connect(self.copy_image_path)
        self.copy_button.setEnabled(True)  # Disable copy button initially
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.color_frame)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.color_frame.setLayout(QHBoxLayout())
        self.color_frame.layout().addWidget(self.status_label)
        self.color_frame.layout().setSpacing(0)
        self.color_frame.layout().setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
    def call_layout(self):
        return self.layout   
    def select_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly  
        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png)", options=options)
        if image_path:
            self.image_path = image_path
            self.image_path_input.setText(image_path)  
            self.copy_button.setEnabled(True)  
        else:
            pass  
    def copy_image_path(self):
        image_path = self.paste_path_input.text().strip()
        if image_path:
            clipboard = QApplication.clipboard()
            clipboard.setText(image_path)
            message_box = QMessageBox()
            message_box.setIcon(QMessageBox.Information) 
            message_box.setWindowTitle("Image Path Copied")
            message_box.setText(f"The image path '{image_path}' has been copied to your clipboard.\n\n"
                            "You can now use it in other applications.")
            message_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Close)
            message_box.setDefaultButton(QMessageBox.Close) 

            message_box.exec_()
            # QMessageBox.information(self, "Image Path Copied", "The image path has been copied to your clipboard.")
        else:
            QMessageBox.warning(self, "No Image Path", "Please paste a valid image path before copying.")

    def capture_image(self):
        # Implement your image capture logic here
        # This part might involve using libraries or system calls depending on your specific requirements
        # For example, you could use OpenCV or the screencapture command on macOS/Linux
        # Once you have the image path, update self.image_path and potentially display the image in the widget
        # ...
        self.image_path = r"C:\Users\a\yolo13\ultralytics1\new_result.bmp"  # Replace with the actual path from your capture logic
        self.copy_button.setEnabled(True)  # Enable copy button after capture

    def get_color(self):
        if self.status == 'NG':
            # return QColor(255, 0, 0)  # Red
            return "#FF6347"  # Red
        elif self.status == 'OK' : 
            return "#32CD32"  # Green
        else:
            return "gray"


class InferenceWorker(QThread):
    progress_updated = pyqtSignal(int)
    inference_completed = pyqtSignal(dict, bytes)

    def save_im(self, data, file_name):
    # Ensure unique filename
        base, ext = os.path.splitext(file_name)
        counter = 1
        while os.path.exists(file_name):
            file_name = f"{base} ({counter}){ext}"
            counter += 1

        # Save data to file
        with open(file_name, 'wb') as file:
            if isinstance(data, Image.Image):
                data.save(file, format="PNG")
            else:
                raise ValueError("Unsupported data type for saving.")

    def save_js(self, data, file_name):
        # Ensure unique filename
        base, ext = os.path.splitext(file_name)
        counter = 1
        while os.path.exists(file_name):
            file_name = f"{base} ({counter}){ext}"
            counter += 1

        # Save data to JSON file
        with open(file_name, 'w') as file:
            json.dump(data, file, indent=4)

    def __init__(self, model, image_path):
        super().__init__()
        self.model = model
        self.image_path = image_path

    def run(self):
        progress = 0
        io_buffer = io.BytesIO()

        final_image, prediction = get_status(self.model, self.image_path)
        print(prediction)
        # final_image.show()
        # Save final_image to file
        self.save_im(final_image, r'output\final_image.png')

        # Save prediction to JSON file
        self.save_js(prediction, r'output\prediction.json')

        final_image.save(io_buffer, format="PNG")
        image_data = io_buffer.getvalue()
        
        # Update the progress bar while performing inference
        while progress < 100:
            progress += 1
            self.progress_updated.emit(progress)
            self.msleep(10)  # Adjust this to match your inference time

        # Emit the signal with the prediction results and image data
        self.inference_completed.emit(prediction, image_data)
        
class YoloResultWindow(QWidget):
    def __init__(self):
        super(YoloResultWindow, self).__init__()
        # start=time.time()
        self.model=YOLO(r'.\best_ver10.pt')  
        self.perform_warmup_inference()
        # print(time.time()-start)
        # io_buffer = io.BytesIO()
        self.image_label = QLabel()
        self.result_boxes_layout = QGridLayout()
        # border_width = 5
        # border_color = Qt.blue
        # bordered_pixmap = apply_border(image_path, border_width, border_color)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)
        # Replace with your actual YOLO prediction processing and image path
        self.prediction = {'PARTS': 'STATUS','BIG FUSE': ' ' , 'SMALL FUSE': ' ' , 'HST': ' ', 'SEAL': ' ', 'LEVER': ' '} #{'PARTS': 'STATUS','BIG FUSE': 'OK', 'SMALL FUSE': 'OK', 'HST YELLOW': 'NG', 'HST BLACK': 'OK'}
        # self.prediction = get_status(model,ima)
        # Save image to BytesIO
        # final_image.save(io_buffer, format="PNG")
        # self.image_data = io_buffer.getvalue()
        self.test_status_label = QLabel("Test  INVALID ")
        self.test_status_label.setStyleSheet("font-weight: bold; font-size: 16px; text-align: center; margin-top: 10px;background-color: gray;")
        self.test_status_label1 = QLabel("Test  INVALID ")
        self.test_status_label1.setStyleSheet("font-weight: bold; font-size: 16px; text-align: center; margin-top: 10px;background-color: gray;")
        self.image_path = r".\noimage.png" # any no image 
        self.paste_path_input = None
        self.main_layout = QHBoxLayout()
        self.main_layout2 = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)  # Set the range of the progress bar
        self.progress_bar.setVisible(False)  # Make progress bar initially invisible
        self.setup_ui()

    def capture_image_from_camera():
        cap = cv2.VideoCapture(0)  # Open the default camera (index 0)
        if not cap.isOpened():
            print("Failed to access the camera.")
            return None
        ret, frame = cap.read()  # Capture frame-by-frame
        cap.release()  # Release the camera
        if not ret:
            print("Failed to capture image.")
            return None
        # Convert the OpenCV frame (BGR format) to QImage
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
        # Convert QImage to QPixmap (optional, depending on how you use it in your application)
        pixmap = QPixmap.fromImage(q_image)
        return pixmap  # Return the QPixmap object

    def perform_warmup_inference(self):
        dummy_image_path = r'.\fuse swapping light off 3.bmp'  
        self.model(dummy_image_path)

    def adjust_window_size(self):
        screen_rect = QDesktopWidget().screenGeometry()
        screen_height = int(screen_rect.height()*(3/4))
        window_width = 500  # Set the width you want here
        self.setGeometry(100, 100, window_width, screen_height)

    def capture_image(self):
        frame = self.capture_image_from_camera()
        
        if frame is None:
            return

        image_data=Image.fromarray(frame[..., ::-1])
        # image_data.show()
        # Show the progress bar
        self.progress_bar.setVisible(True)
        # Create and start the inference worker thread
        self.inference_thread = InferenceWorker(self.model, image_data)
        self.inference_thread.progress_updated.connect(self.update_progress_bar)
        self.inference_thread.inference_completed.connect(self.display_results)
        self.inference_thread.start()
        
        
    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def display_results(self, prediction,image_data):
        self.prediction = prediction
        
        image_pixmap = QPixmap()
        image_pixmap.loadFromData(image_data, format="PNG")
        self.image_label.setPixmap(image_pixmap.scaled(640, 500))
        self.image_label.setStyleSheet("border: 5px solid #3C0008; border-radius: 20px;")
        
        row = 0
        count = 0
        for label, status in self.prediction.items():
            result_box = ResultBox(label, status, count, self.prediction)
            count += 1
            self.result_boxes_layout.addWidget(result_box, row, 0)
            row += 1

            if len(self.prediction) == count:
                layout_prev = result_box.call_layout()

                label_style1 = 'font-size: 15pt;text-align: center;font-weight: bold;color: #800000'
                label_style = 'font-size: 15pt;text-align: center;font-weight: bold'
                self.spacer = QWidget()

                self.test_status_layout = QHBoxLayout()
                self.test_status_layout.setAlignment(Qt.AlignCenter)
                self.test_label = QLabel("RESULT : ")
                self.test_label.setStyleSheet(label_style1)
                self.test_status_layout.addWidget(self.test_label, alignment=Qt.AlignRight)

                is_all_ok = all(value == "OK" for key, value in self.prediction.items() if key != 'PARTS')
                if is_all_ok:
                    test_status = "OK"
                    color1 = "green"
                else:
                    test_status = "NG"
                    color1 = "red"

                self.status_label = QLabel(test_status)
                self.status_label.setStyleSheet(f"{label_style}; color: {'green' if test_status == 'OK' else 'red'};")
                self.test_status_layout.addWidget(self.status_label, stretch=0)

                self.test_status_box = QFrame()
                self.test_status_box.setStyleSheet("border-radius: 5px; background-color: lightgoldenrodyellow; width: 30px; height: 30px;")
                self.test_status_box.setLayout(self.test_status_layout)
                self.result_boxes_layout.addWidget(self.test_status_box, row, 0)

                layout_prev.addLayout(self.result_boxes_layout)
                self.setLayout(layout_prev)

        self.progress_bar.setVisible(False)

    def capture_image_from_camera(self):
        cap = cv2.VideoCapture(0)  # 0 is usually the default camera
        if not cap.isOpened():
            QMessageBox.critical(self, "Camera Error", "Failed to access the camera.")
            return None

        ret, frame = cap.read()
        cap.release()
        if not ret:
            QMessageBox.critical(self, "Capture Error", "Failed to capture image.")
            return None

        return frame

    def setup_ui(self):
        spacer_left = QWidget()
        spacer_right = QWidget()

        self.test_status_label.setText("TEST ")
        self.test_status_label.setStyleSheet("font-weight: bold; "  # Optional styling
                                         "color: black; "  # Ensure clear text color
                                         "TEST INVALID { background-color: lightgray; }")
                                         
        self.test_status_label1.setText("VALID")
        self.test_status_label1.setStyleSheet("font-weight: bold; "  # Optional styling
                                         "color: black; "  # Ensure clear text color
                                         "TEST INVALID { background-color: lightgray; }")

        self.image_pixmap = QPixmap(self.image_path)
        # self.image_pixmap = QPixmap()
        # self.image_pixmap.loadFromData(self.image_data, format="PNG")
        self.image_label.setPixmap(self.image_pixmap.scaled(640,500))
        self.image_label.setStyleSheet("border: 5px solid #3C0008; border-radius: 20px;")
        
        row = 0
        for label, status in self.prediction.items():
            result_box = ResultBox(label, status ,1000,self.prediction)
            self.result_boxes_layout.addWidget(result_box, row, 0)
            row +=1
        #############################
        self.main_layout.capture_button = QPushButton("CAPTURE")
        self.main_layout.capture_button.setStyleSheet("""
             QPushButton {
        background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(255, 204, 102, 1), stop:1 rgba(255, 165, 0, 1));
        border-radius: 11px;
        color:#800000;
        margin: 11px;
        font-weight: bold; 
        width: 90px; height: 31px; 
                      }
             QPushButton:hover {
        background-color: #FFE000;
                 }
            QPushButton:pressed {
        background-color: #FAFAD2;
               }
                                """)
        self.main_layout.capture_button.clicked.connect(self.capture_image)  

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)  # Set the range of the progress bar

        self.main_layout.addItem(QSpacerItem(50,500, QSizePolicy.Minimum, QSizePolicy.Preferred))  # Adjust spacer size
        self.main_layout.addWidget(self.progress_bar)  # Add progress bar below buttons
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.main_layout.capture_button)
        # self.main_layout.addWidget(self.main_layout.copy_button)
        # self.main_layout.addWidget(self.main_layout.paste_path_input) 
        #self.main_layout.addWidget(self.main_layout.capture_button)

        # self.main_layout.browse_button = QPushButton("Browse")
        # self.main_layout.browse_button.clicked.connect(self.browse_image)
        # self.main_layout.addWidget(self.main_layout.browse_button)
       
        #self.main_layout.addWidget(self.main_layout.paste_path_input) 
        
        # main_layout2 = QVBoxLayout()
        self.main_layout2.addWidget(self.image_label)
        self.main_layout2.addLayout(self.result_boxes_layout)
        # self.main_layout2.addLayout(self.test_status_label)
        ##########################################
        
        self.adjust_window_size()                                                                                
        self.main_layout2.addLayout(self.main_layout)  
        
        # style_string = "font-size: 24pt; text-align: center;" 
        style_string =""" border-radius: 10px;
                 background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(255, 204, 102, 1), stop:1 rgba(255, 165, 0, 1));
                 color: #800000;font-weight: bold;padding: 5px 10px;"""
        title_label = QLabel('IP FUSE BOX PART DETECTION')
        
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(style_string)
        layout = QVBoxLayout()
        layout.addWidget(title_label)
        layout.addLayout(self.main_layout2)
        self.setLayout(layout)
        self.setStyleSheet("background-color: lightgoldenrodyellow")
        
        self.setWindowTitle('Part Detection UI')
        self.setStyleSheet("background-color: lightgoldenrodyellow")
        self.show()

if __name__ == '__main__':
    app= 0
    app = QApplication(sys.argv)
    window = YoloResultWindow()
    sys.exit(app.exec_())