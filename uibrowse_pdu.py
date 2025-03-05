import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,QFormLayout,QSpacerItem,QSizePolicy,
                             QLabel, QFrame, QGridLayout,QPushButton,QFileDialog,QMessageBox,QLineEdit,QFontComboBox,QProgressBar,
                             QDesktopWidget)
from PyQt5.QtGui import QPixmap, QColor ,QPalette, QMovie
from PyQt5.QtCore import Qt,QThread,pyqtSignal,QObject,QTimer
import os
from pdulive import get_status
from ultralytics import YOLO
import io
import torch
from PIL import Image
from torchvision import transforms
import time
  
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
        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp)", options=options)
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

    def __init__(self, model, image_path):
        super().__init__()
        self.model = model
        self.image_path = image_path

    def run(self):
        
        
        progress = 0
        io_buffer = io.BytesIO()
        # start_time = time.time()
        # Perform inference using your existing logic
        final_image, prediction = get_status(self.model, self.image_path)
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
        #start=time.time()
        self.model=YOLO(r'C:\Users\jovin\1.ML\ultralytics\best_ver10.pt') 
        self.model.to('cuda')  # Move the model to GPU 
        self.perform_warmup_inference()
        #print(time.time()-start)
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
        self.image_path = r"C:\Users\jovin\1.ML\ultralytics\noimage.png" # any no image 
        self.paste_path_input = None
        self.main_layout = QHBoxLayout()
        self.main_layout2 = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)  # Set the range of the progress bar
        self.progress_bar.setVisible(False)  # Make progress bar initially invisible
        self.setup_ui()

    def perform_warmup_inference(self):
        dummy_image_path = r'C:\Users\jovin\1.ML\ultralytics\Image__2024-05-22__15-45-04.jpg'
        
        # Load and preprocess the image
        image = Image.open(dummy_image_path).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize((640, 640)),  # Resize to model input size
            transforms.ToTensor(),  # Convert to tensor
        ])
        image_tensor = preprocess(image).unsqueeze(0).to('cuda')  # Add batch dimension and move to GPU

        # Perform inference
        with torch.no_grad():
            self.model(image_tensor)

    def adjust_window_size(self):
        screen_rect = QDesktopWidget().screenGeometry()
        screen_height = int(screen_rect.height()*(3/4))
        window_width = 500  # Set the width you want here
        self.setGeometry(100, 100, window_width, screen_height)

    def capture_image(self):
        # ...
        image_path1 = self.main_layout.paste_path_input.text().strip() # Replace with the actual path from your capture logic
        # print("captured image",image_path1)
        ###self.main_layout.capture_button.setEnabled(True)
        if not image_path1:
            QMessageBox.warning(self, "No Image Path", "Please paste a valid image path.")
            return
        
        # Show the progress bar
        self.progress_bar.setVisible(True)
        # Create and start the inference worker thread
        self.inference_thread = InferenceWorker(self.model, image_path1)
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

                # Now, add the "TEST OK/NG" widget to the new layout (which is QVBoxLayout)
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

        # Hide the progress bar
        self.progress_bar.setVisible(False)

    def copy_image_path(self):
        image_path = self.main_layout.paste_path_input.text().strip()
        self.image_pixmap = QPixmap(image_path)
        
        self.image_label.setPixmap(self.image_pixmap.scaled(640,500))
        self.image_label.setStyleSheet("border: 5px solid #3C0008; border-radius: 20px;")
        # print(image_path)
        if image_path:
             extension = os.path.splitext(image_path)[1].lower()
             if extension in (".jpg", ".jpeg", ".bmp" , ".png"):
                 clipboard = QApplication.clipboard()
                 clipboard.setText(image_path)
                 QMessageBox.information(self, "Image Path Copied", "The image path has been copied.")
             else:
                 QMessageBox.warning(self, "No Image Path", "Please paste a valid image path before copying.")
        else:
                 QMessageBox.warning(self, "No Image Path", "Please paste a valid image path before copying.")
        self.prediction = {'PARTS': 'STATUS','BIG FUSE': ' ', 'SMALL FUSE': ' ', 'HST': ' ', 'SEAL': ' ', 'LEVER': ' '} #{'PARTS': 'STATUS','BIG FUSE': 'OK', 'SMALL FUSE': 'OK', 'HST YELLOW': 'NG', 'HST BLACK': 'OK'}
        row = 0
        count=0
        for label, status in self.prediction.items():
            
            result_box = ResultBox(label, status, count, self.prediction)
            # print(f"{count}:{result_box}")
            count=count + 1
            self.result_boxes_layout.addWidget(result_box, row, 0)
            row +=1
            if len(self.prediction) == count  :
                layout_prev= result_box.call_layout()
                
                #############################
                

        # Now, add the "TEST OK/NG" widget to the new layout (which is QVBoxLayout)
                    #   self.layout.addWidget(self.test_status_box)
                ##############################
                label_style1= 'font-size: 15pt;text-align: center;font-weight: bold;color: #800000'
                label_style= 'font-size: 15pt;text-align: center;font-weight: bold'
                self.spacer = QWidget()
 #    self.label_and_status_layout.addWidget(self.spacer, stretch=1)  # Adjust stretch for centering

 # Create layout for test status (OK/NG)
                self.test_status_layout = QHBoxLayout()
                self.test_status_layout.setAlignment(Qt.AlignCenter)
                # self.test_status_layout.setSpacing(0) # Remove spacing between elements
                # self.test_status_layout.setContentsMargins(0, 0, 0, 0) # Remove margins
 # Create label for test status text ("TEST:")
                self.test_label = QLabel("RESULT : ")
                self.test_label.setStyleSheet(label_style1)
                self.test_status_layout.addWidget(self.test_label,alignment=Qt.AlignRight)


#                 is_all_ok = all(value == "OK" for key, value in self.prediction.items() if key != 'PARTS')
#                 if is_all_ok:
#                   test_status= "OK"
#                   color1 = "green"
#                 else :
#                   test_status= "NG"
#                   color1 = "red"
#  # Create label for OK/NG status (colored)
                self.status_label = QLabel(" ")

                self.status_label.setStyleSheet(f"{label_style}; "
                                                 "color: 'gray'")
                self.test_status_layout.addWidget(self.status_label,stretch=0)
        
                self.test_status_box = QFrame()
                self.test_status_box.setStyleSheet("border-radius: 5px; background-color: lightgoldenrodyellow; width: 30px; height: 30px;") # Adjust styling as needed
                self.test_status_box.setLayout(self.test_status_layout)
                # self.test_status_box.setMinimumSize(30, 30)
                # self.test_status_box.setFixedWidth(60)
    #             temporary_widget = QWidget()
                self.result_boxes_layout.addWidget(self.test_status_box, row, 0)
                # self.result_boxes_layout.setColumnStretch(row, 0) 
    # # Set the temporary widget's layout to the new QVBoxLayout
    # #             temporary_widget.setLNG" widget) to the existing QHBoxLayout (layout_prev)
                layout_prev.addLayout(self.result_boxes_layout) 
                # # layout_prev.addWidget(self.test_status_box,row+1,0)
                self.setLayout(layout_prev)

    def setup_ui(self):
        spacer_left = QWidget()
        spacer_right = QWidget()

# Add spacer widgets and label to grid (adjust row/column as needed)
        
        self.test_status_label.setText("TEST ")
        self.test_status_label.setStyleSheet("font-weight: bold; "  # Optional styling
                                         "color: black; "  # Ensure clear text color
                                         "TEST INVALID { background-color: lightgray; }")
                                         
        self.test_status_label1.setText("VALID")
        self.test_status_label1.setStyleSheet("font-weight: bold; "  # Optional styling
                                         "color: black; "  # Ensure clear text color
                                         "TEST INVALID { background-color: lightgray; }")

        # self.test_status_label = QLabel("TEST INVALID")
        # self.test_status_label.setStyleSheet("font-weight: bold; font-size: 16px; text-align: center; margin-top: 10px;background-color: green;")

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
        self.main_layout.paste_path_input = QLineEdit()
        self.main_layout.paste_path_input.setPlaceholderText("Paste image path here")

        # Add Copy Path button
        self.main_layout.copy_button = QPushButton("Copy Path")
        self.main_layout.copy_button.setStyleSheet("""
             QPushButton {
        background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(255, 204, 102, 1),
                                stop:1 rgba(255, 165, 0, 1));
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
        self.main_layout.copy_button.clicked.connect(self.copy_image_path)
        self.main_layout.copy_button.setEnabled(True)
        #self.main_layout.addWidget(self.main_layout.copy_button)  # Add Copy Path button

        self.main_layout.capture_button = QPushButton("Start")
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
        self.main_layout.capture_button.clicked.connect(self.capture_image)  # Implement capture logic

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)  # Set the range of the progress bar

        self.main_layout.addItem(QSpacerItem(50,500, QSizePolicy.Minimum, QSizePolicy.Preferred))  # Adjust spacer size
        self.main_layout.addWidget(self.progress_bar)  # Add progress bar below buttons
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.main_layout.capture_button)
        self.main_layout.addWidget(self.main_layout.copy_button)
        self.main_layout.addWidget(self.main_layout.paste_path_input) 
        #self.main_layout.addWidget(self.main_layout.capture_button)

        # Add Browse button
        self.main_layout.browse_button = QPushButton("Browse")
        self.main_layout.browse_button.clicked.connect(self.browse_image)
        self.main_layout.addWidget(self.main_layout.browse_button)
                
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

    def browse_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp)", options=options)
        if image_path:
            self.main_layout.paste_path_input.setText(image_path)
# class workerthread(QThread):
#     status_progress=pyqtSignal(str)

if __name__ == '__main__':
    app= 0
    app = QApplication(sys.argv)
    window = YoloResultWindow()
    sys.exit(app.exec_())