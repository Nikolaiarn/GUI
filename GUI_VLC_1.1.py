
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, 
QSlider, QSizePolicy, QFrame, QGraphicsScene, QGraphicsView, QComboBox, QAction, QWidgetAction, QSpacerItem)
from PyQt5.QtGui import QColor, QPixmap, QPainter, QPen, QImage, QPalette, QLinearGradient
from PyQt5.QtCore import Qt, QTimer, QDateTime, pyqtSlot
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PyQt5 import QtWidgets
from os import listdir
from os.path import isfile, join
from cv2 import VideoCapture, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT
import vlc
from collections import defaultdict, Counter

class ImageWindow(QMainWindow):
    def __init__(self, img):
        super().__init__()
        self.setWindowTitle("Unfiltered Frame")

        self.img = img  # save the image for later saving

        # Convert the OpenCV image format to QPixmap
        qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimg)

        # Create QGraphicsScene and add QPixmap to it
        self.scene = QGraphicsScene(self)
        self.scene.addPixmap(pixmap)

        # Create GraphicsView and set its scene
        self.view = GraphicsView(self.scene)

        # Create "Save Image" button
        self.save_button = QPushButton("Save Image", self)
        self.save_button.clicked.connect(self.save_image)

        # Set GraphicsView and QPushButton as central widgets
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addWidget(self.save_button)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.show()

    @pyqtSlot()
    def save_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png)")
        if file_path:
            cv2.imwrite(file_path, self.img)


class GraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super(GraphicsView, self).__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)

    def wheelEvent(self, event):
        # Zoom Factor
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor

        # Set Anchors
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # Handle Wheel Event
        if event.angleDelta().y() > 0:
            zoomFactor = zoomInFactor
        else:
            zoomFactor = zoomOutFactor
        self.scale(zoomFactor, zoomFactor)


#Creating a class for making the component visualization
class DetectionsTimeline(QWidget):
    frame_count = 1
    component_names = ['Anchor', 'Buoy', 'Chain', 'Fiber thimple', 'H-link', 'Rope', 'Shackle', 'Triplate', 'Wire', 'Wire socket', 'Components']

    def __init__(self, parent= "VideoPlayerWindow"):
        # Initializing colors, component number dictionary and layout
        super().__init__(parent)
        self.component_colors = {
            'Chain': QColor(255, 153, 51),
            'Wire': QColor(76, 153, 0),
            'Rope': QColor(255, 192, 105),
            'Triplate': QColor(153, 255, 153),
            'H-link': QColor(211, 242, 97),
            'Shackle': QColor(178, 255, 102),
            'Buoy': QColor(255, 153, 153),
            'Fiber thimple': QColor(9, 109, 217),
            'Anchor': QColor(173, 198, 255),
            'Wire socket': QColor(102, 255, 255),
            'Components': QColor(255, 0, 0)
        }

        self.detections = {}
        self.frame_count = 0
        self.current_frame = 0

        #Visualization of timeline
        frame_count = 0
        self.layout = QVBoxLayout(self)  # Set the layout for the widget
        self.set_frame_count(frame_count)
        self.setAttribute(Qt.WA_PaintOnScreen)
        self.color_legend_layout = QVBoxLayout()
        self.layout.addLayout(self.color_legend_layout)
        self.color_legend_labels = []

    def set_frame_count(self, frame_count):
        self.frame_count = frame_count
        self.update()
        print(f'Set frame count: {frame_count}')
       
    # updating visualization of the timeline 
    def set_detections(self, detections):  
        self.detections = detections
        self.update()  # update will cause the widget to be repainted

    def get_color_for_component(self, component):
        return self.component_colors.get(component, QColor(2, 230, 240))  # default to white color if component not found

    def paintEvent(self, event):        #Painting the component timeline
        painter = QPainter(self)

        pos = 100
        back_side = 85
        space_size = 30
        vertical_offset = 15

        # Calculate the width of one frame in pixels
        frame_width = (self.width() - pos - back_side) / (self.frame_count + 1)
        detect_dic = {}
        counter = 0
        
        # Draw colored rectangles for each component detected in each frame
        for n, (component, frames) in enumerate(self.detections.items()):
            detect_dic[component] = frames
        
        for comp in self.component_names:
            if comp in detect_dic:
                i = counter
                color = self.get_color_for_component(comp)
                for frame in detect_dic[comp]:
                    painter.setPen(QPen(color, 1))
                    painter.drawLine(int((frame * frame_width) + pos), int(i * space_size + vertical_offset), int((frame * frame_width) + pos), int((i * space_size) + vertical_offset + 10))
                counter += 1
               
# creating main GUI window, displaying video etc.
class VideoPlayerWindow(QMainWindow):
    component_names = ['Anchor', 'Buoy', 'Chain', 'Fiber thimple', 'H-link', 'Rope', 'Shackle', 'Triplate', 'Wire', 'Wire socket']
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dag-Børre's Video Player")
        self.setGeometry(100, 100, 800, 600)
        self.create_menus()
        self.create_layout()
        self.create_video_viewer()
        self.create_video_controls()
        self.setup_timeline()
        self.initialize_variables()
        self.adjust_video_speed(1)
        
        self.raw_frame_button = QPushButton("#No filter (Image)", self)
        self.raw_frame_button.clicked.connect(self.show_raw_frame)
        self.raw_frame_button.hide()
        self.player_layout.addWidget(self.raw_frame_button)

        self.show()

    def create_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        # Create 'Upload Folder' action
        upload_folder_action = QAction('Upload Folder', self)
        upload_folder_action.triggered.connect(self.upload_folder)
        file_menu.addAction(upload_folder_action)

        # Create 'Upload Video' action
        upload_video_action = QAction('Upload Video', self)
        upload_video_action.triggered.connect(self.upload_video)
        file_menu.addAction(upload_video_action)

        # Create 'Upload JSON' action
        upload_json_action = QAction('Upload JSON', self)
        upload_json_action.triggered.connect(self.upload_json)
        file_menu.addAction(upload_json_action)

        # Set the geometry and stylesheet of the menubar
        menubar.setGeometry(0, 0, self.width(), menubar.height())  
        menubar.setStyleSheet("QMenuBar{spacing: 100px;}") 
       
        layout = QHBoxLayout(self)
        self.time_label = QLabel("00:00 / 00:00     ", self)
        menubar.setCornerWidget(self.time_label, Qt.TopRightCorner)
       
        # Set the menubar of the window
        self.setMenuBar(menubar)

    def create_layout(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.video_control_layout = QVBoxLayout()
        self.layout.addLayout(self.video_control_layout)

    def create_video_viewer(self):
        self.instance = vlc.Instance('--noaudio')
        self.media_player = self.instance.media_player_new()
        fps = self.media_player.get_fps()
        fps = round(fps, 3)
        print(f'Fps: {fps}')
        self.video_view = QWidget(self)
        self.media_player.set_hwnd(self.video_view.winId())
        self.video_control_layout.addWidget(self.video_view)
        self.video_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Expandable video

    def create_video_controls(self):
        self.player_frame = QFrame(self)
        self.player_layout = QHBoxLayout(self.player_frame)
        self.video_control_layout.addWidget(self.player_frame)

        self.start_pause_button = QPushButton("Start Playback", self)
        self.start_pause_button.clicked.connect(self.toggle_playback)
        self.player_layout.addWidget(self.start_pause_button)

        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setTickPosition(QSlider.TicksAbove)
        self.timeline.setTickInterval(1)
        self.player_layout.addWidget(self.timeline)

        self.video_speed_combobox = QComboBox(self)
        self.video_speed_combobox.addItems(["1x", "2x", "4x", "8x", "16x"])
        self.video_speed_combobox.currentIndexChanged.connect(self.adjust_video_speed)
        self.player_layout.addWidget(self.video_speed_combobox)

    def setup_timeline(self):
        self.visual_timeline = DetectionsTimeline(self)
        self.layout.addWidget(self.visual_timeline)

    def initialize_variables(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_video)
        self.video_path = None
        self.frame_count = 0
        self.current_frame = 0
        self.detections = {}
        self.last_frame_time = QDateTime.currentDateTime()
    
    def process_video(self, video_path):
        self.media = self.instance.media_new(video_path)
        self.media.parse()
        self.media_player.set_media(self.media)
        fps = self.media_player.get_fps()
        frame_duration = round(1000 / fps, 3)
        print(frame_duration)
        self.frame_count = int(self.media.get_duration() // frame_duration) # !!!!!!!!!!!Get duration in seconds(Might be wrong to do this; should be 1000)
        self.timeline.setMinimum(0)
        self.timeline.setMaximum(self.frame_count - 1)
        self.timeline.setValue(0)
        self.timeline.valueChanged.connect(self.update_frame)
        self.visual_timeline.set_frame_count(self.frame_count)
        self.timer.timeout.connect(self.update_video)  # Connect timer's timeout signal to update_video method

        # Calculate the total time in minutes and seconds
        total_time_seconds = self.frame_count / self.media_player.get_fps()
        total_minutes, total_seconds = divmod(total_time_seconds, 60)
        # Set the total time
        self.total_time = f"{int(total_minutes):02d}:{int(total_seconds):02d}"

        # Start the timer and set its interval to the frame duration
        self.timer.start(int(frame_duration))

        self.video_path = video_path  # store the video path in the instance variable
        self.raw_frame_button.setEnabled(True)  # enable the button when a video is loaded
       

    def update_frame(self, frame_idx):
        fps = self.media_player.get_fps()
        frame_duration = int(1000 / fps)
        self.current_frame = frame_idx
        self.timeline.setValue(self.current_frame)
        self.media_player.set_time(frame_idx * frame_duration)  

    
    # Uploading video and changing button layout if video uploaded
    def upload_video(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi *.vob *.mpg)")
        video_path, _ = file_dialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.vob *.mpg)")

        if video_path:
            self.process_video(video_path)
            self.sender().setText('Video Uploaded') 

    def update_video(self):
        fps = self.media_player.get_fps()
        frame_duration = int(1000 / fps)
        if self.media_player.get_state() == vlc.State.Ended:
            self.media_player.stop()  # Stop the video
            self.media_player.play()  # Start the video from the beginning
            self.current_frame = 0  # Reset current frame to the beginning
            self.timeline.setValue(self.current_frame)  # Reset timeline to the beginning
            
        else:
            self.current_frame = self.media_player.get_time() // frame_duration
            self.timeline.blockSignals(True)  # Block signals to prevent feedback loop
            self.timeline.setValue(self.current_frame)
            self.timeline.blockSignals(False)  # Unblock signals

        # Calculate the current time in minutes and seconds
        current_time_seconds = self.current_frame / self.media_player.get_fps()
        current_minutes, current_seconds = divmod(current_time_seconds, 60)

        # Update the time label
        self.time_label.setText(f"{int(current_minutes):02d}:{int(current_seconds):02d} / {self.total_time}     ")

        
    def upload_folder(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        folder_path = dialog.getExistingDirectory(self, "Select Folder")

        if folder_path:
            video_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith((".mp4", ".avi", ".vob", ".mpg"))]
            json_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith(".json")]

            if video_files and json_files:
                video_path = join(folder_path, video_files[0])
                json_path = join(folder_path, json_files[0])

                self.process_video(video_path)
                self.handle_json_selected(json_path)
                self.sender().setText('Folder Uploaded')
    
    @pyqtSlot()
    def toggle_playback(self):
        if self.media_player.is_playing():
            self.media_player.pause()
            self.start_pause_button.setText("Start")
            self.raw_frame_button.show()
        else:
            self.media_player.play()
            self.start_pause_button.setText("Pause")
            self.last_frame_time = QDateTime.currentDateTime()
            self.raw_frame_button.hide()
            

    def adjust_video_speed(self, index):
        speed_options = [1, 2, 4, 8, 12]
        selected_speed = int(speed_options[int(index)])
        self.video_speed = selected_speed
        self.media_player.set_rate(self.video_speed)  # Set the rate (speed) of the media player'

    @pyqtSlot()
    def show_raw_frame(self):
        raw_video_path = self.video_path.replace(".mp4", "_Raw.mp4")
        raw_cap = cv2.VideoCapture(raw_video_path)
        raw_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = raw_cap.read()
        if ret:
            self.image_window = ImageWindow(frame)
        raw_cap.release()

        

#____________________________________________________________JSON______________________________________________________________________________________________

# Detects information from the JSON file, and gives the different parameters the correct values
    def load_detections(self, filename):
        label_map = ['Anchor', 'Buoy', 'Chain', 'Fiber thimple', 'H-link', 'Rope', 'Shackle', 'Triplate', 'Wire', 'Wire socket']
        with open(filename) as f:
            data = json.load(f)

        # Create a dictionary where the keys are the component names and the values are Counters of frame numbers.
        detections = defaultdict(Counter)
        for item in data:
            frame_number = item["frame_number"]
            label = label_map[int(item["label"])]
            bbox = item["x_min"], item["y_min"], item["x_max"], item["y_max"]

            detections[label][frame_number] += 1

        # Create a copy of detections before any removals
        initial_detections = {k: sum(v.values()) for k, v in detections.items() if sum(v.values()) > 0}

        # Create a new dictionary with components that have more than 20 frames
        detections = {k: v for k, v in detections.items() if sum(v.values()) > 20}

        # for the remaining components, check the frame count in the range using sliding window
        to_remove = set()
        remove_counter = defaultdict(int) # count number of frames removed for each label

        for label, frames in detections.items():
            sorted_frames = sorted(frames.elements())
            start_index = 0
            end_index = 0
            window_start = 0
            window_end = 100

            while window_end <= self.frame_count:
                while end_index < len(sorted_frames) and sorted_frames[end_index] < window_end:
                    end_index += 1
                
                window_frames = sorted_frames[start_index:end_index]
                window_frame_count = len(window_frames)
                
                if window_frame_count < 5:
                    # if less than 5 occurrences within the window, mark for removal
                    for frame in window_frames:
                        to_remove.add((label, frame))
                        remove_counter[label] += 1  # increment counter

                # slide the window forward by 50 frames
                window_start += 50
                window_end += 50
                while start_index < len(sorted_frames) and sorted_frames[start_index] < window_start:
                    start_index += 1

        # remove frames that have less than 5 nearby occurrences
        for label, frame in to_remove:
            detections[label][frame] -= 1
            if detections[label][frame] <= 0:
                del detections[label][frame]

        # remove components that have no frames left
        detections = {k: v for k, v in detections.items() if v}

        # Print the number of frames removed for each label
        for label, initial_count in initial_detections.items():
            final_count = sum(detections[label].values()) if label in detections else 0
            removed_count = initial_count - final_count
            print(f"Start_{label}: {initial_count} | End_{label}: {final_count} | Removed: {removed_count}")
        
        # Print a line of dashes after the labels
        print("-------------------------")

        return detections
    
        # Uploads json file, updates detections in timeline, deletes previous timelines
    def upload_json(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("JSON Files (*.json)")
        file_dialog.fileSelected.connect(self.handle_json_selected)
        file_dialog.exec_()

    def handle_json_selected(self, json_path):
        if json_path:
            detect_dic = {}
            self.detections = self.load_detections(json_path)
            self.load_detections(json_path)

            self.visual_timeline.set_detections(self.detections)  # Update the detections in timeline

            for label in self.visual_timeline.color_legend_labels:
                label.deleteLater()
            self.visual_timeline.color_legend_labels.clear()

            for n, (component, frames) in enumerate(self.detections.items()):
                detect_dic[component] = frames

            # Create labels and timelines after the JSON file is read
            for component in self.component_names:
                if component in self.detections:  # Check if the component is in the JSON file
                    color = self.visual_timeline.get_color_for_component(component)
                    label = QLabel(component)
                    self.visual_timeline.color_legend_layout.addWidget(label)
                    self.visual_timeline.color_legend_labels.append(label)
            self.visual_timeline.update()


if __name__ == '__main__':
    app = QApplication([])
    app.setStyle("Fusion")

    palette = app.palette()
    palette.setColor(palette.Window, QColor(240, 230, 240))
    app.setPalette(palette)

    window = VideoPlayerWindow()
    window.adjust_video_speed(1.0)  # Set the initial video speed
    window.show()

    app.exec_()