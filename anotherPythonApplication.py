from darkflow.net.build import TFNet
import cv2
import sys
import numpy as np
# import matplotlib.pyplot as plt
# from tkinter import *
# from PIL import Image
# from PIL import ImageTk
# import tkFileDialog
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QDesktopWidget
from PyQt5.QtGui import *
from PyQt5.QtCore import *



options = {"model": "cfg/yolo-obj.cfg", "load": "bin/yolo-obj_300.weights", "threshold": 0.72, "saveVideo":""}

tfnet = TFNet(options)

# Set up tracker.
tracker = cv2.MultiTracker_create()
init_once = False

# Read video
video = cv2.VideoCapture("./sample_img/4axismovement.mp4")

# Exit if video not opened.
if not video.isOpened():
    print('Could not open video')
    sys.exit()

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

# Read first frame.
ok, frame = video.read()
if not ok:
    print('Cannot read video file')
    sys.exit()

yield_object_total_count = 0
bounding_boxes_has_changed = 0
is_first_frame = 1
tracking_list = []  # [(x1y1),(x2,y2)] - stores the mid points given from tracker
new_detection = 0
current_detections_on_d_frame = 0
frame_count = 1
detection_x_to_plot=[]
detection_y_to_plot=[]
# initialize the window toolkit along with the two image panels
# root = Tk()
# global panelA, panelB
# panelA = None
# panelB = None
app = QApplication(sys.argv)
# frame2 = QWidget()  # Replace it with any frame you will putting this label_image on it
# label_Image = QLabel(frame2)
while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    same_detections_this_frame = []
    bounding_boxes_for_tracker = []
    current_frame_detection_id = 0
    detection_timer_start=cv2.getTickCount()
    result = tfnet.return_predict(frame)
    detection_timer_end = cv2.getTickCount()
    print(result)
    # Start timer
    timer = cv2.getTickCount()
    for one_detection in result:
        left = one_detection['topleft']['x']
        top = one_detection['topleft']['y']
        right = one_detection['bottomright']['x']
        bot = one_detection['bottomright']['y']
        mid_x = left + (right - left) / 2
        mid_y = top + (bot - top) / 2
        cv2.rectangle(frame, (left, top), (right, bot), (0, 255, 0), 3, 1)
        if mid_y > 113 and mid_y < 599 and mid_x>140 and mid_x < 1119:  # safe upper margin
            # current_frame_detection_id +=1
            same_detection_has_found = 0
            print("-----current_frame_detection_id ------", current_frame_detection_id)
            # detection_x_to_plot.append(mid_x)
            # detection_y_to_plot.append(mid_y)

            if is_first_frame == 1:
                new_detection = 1
            if is_first_frame == 0:
                for index in range(len(tracking_list)):
                    # print('---11111111111-------')
                    print("----(tracking_list)index----", index)
                    detect_and_track_x_diff = abs(mid_x - tracking_list[index][0])
                    detect_and_track_y_diff = abs(mid_y - tracking_list[index][1])
                    print(detect_and_track_x_diff)
                    print(detect_and_track_y_diff)

                    if same_detection_has_found == 0:
                        if detect_and_track_x_diff < 50 and detect_and_track_y_diff < 61:
                            # new_detection = 1;
                            same_detection_has_found = 1
                            print('---same_detection-------')
                            same_detections_this_frame.append(current_frame_detection_id)

            bounding_box = (left, top, right - left, bot - top)
            bounding_boxes_for_tracker.append(bounding_box)
            current_frame_detection_id += 1
    # end of detections loop

    is_first_frame = 0

    print("current_frame_detection_id", current_frame_detection_id)
    print("len(same_detections_this_frame)", len(same_detections_this_frame))
    print(same_detections_this_frame)
    print("len(bounding_boxes_for_tracker)", len(bounding_boxes_for_tracker))
    print(bounding_boxes_for_tracker)

    if len(same_detections_this_frame) > 0:  # has previous detections
        for index1 in sorted(same_detections_this_frame, reverse=True):
            del bounding_boxes_for_tracker[index1]  # removing previous detected boxes
    print("len(bounding_boxes_for_tracker) after", len(bounding_boxes_for_tracker))

    if len(bounding_boxes_for_tracker) > 0:  # has new bboxes to add
        new_detection = 1

    if ok:
        if not init_once:
            if new_detection == 1:
                for index in range(len(bounding_boxes_for_tracker)):
                    ok = tracker.add(cv2.TrackerMIL_create(), frame, bounding_boxes_for_tracker[index])
                    yield_object_total_count += 1
                # init_once = True
                new_detection = 0

        # Update tracker
        ok, boxes = tracker.update(frame)
        print(ok, boxes)

        # reset main tracker on each and every 20 frames
        # if frame_count  == 0:
        if frame_count%20==0 :
            # plt.plot(detection_x_to_plot, detection_y_to_plot)
            # plt.show()
            print("tracker reset")
            keeping_boxes=[]
            tracker = cv2.MultiTracker_create()
            for one_box in boxes:
                print(one_box)
                if one_box[1] + one_box[3]/2 < 600 and one_box[0] + one_box[2]/2 > 135 and one_box[1] + one_box[3]/2 >112 and one_box[0] + one_box[2]/2 <1120: #inside the box
                    bbox_to_add = (one_box[0],
                                   one_box[1],
                                   one_box[2],
                                   one_box[3])
                    ok = tracker.add(cv2.TrackerMIL_create(), frame, bbox_to_add)
                    keeping_boxes.append(one_box)
            boxes=keeping_boxes
            print("filtered boxes",boxes)

        tracking_list = []
        # Draw bounding box
        for newbox in boxes:
            track_mid_x = newbox[0] + newbox[2] / 2
            track_mid_y = newbox[1] + newbox[3] / 2
            tracking_list.append((track_mid_x, track_mid_y))
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, (200, 2, 1), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)



    # Calculate Frames per second (FPS)
    # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    detection_timer_start
    print("time for detecting", (detection_timer_end-detection_timer_start) / cv2.getTickFrequency())
    print("time for tracking",( cv2.getTickCount() - timer)/cv2.getTickFrequency())


    # Display FPS on frame

    cv2.putText(frame, "Frame No : " + str(int(frame_count)), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),
                2)
    # cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    cv2.putText(frame, "Count : " + str(int(yield_object_total_count))+" / Weight : " + str(float(yield_object_total_count*0.123))+"kg", (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (254, 254, 254), 2);
    # medium size tomato weight 123g

    # Draw margins
    cv2.rectangle(frame, (135, 113), (1120, 600), (2, 2, 253), 1, 1)

    print("frame_count", frame_count)
    frame_count += 1

    # Display result
    cv2.imshow("Tracking", frame)

    # OpenCV represents images in BGR order; however PIL represents
    # images in RGB order, so we need to swap the channels
    # count_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # convert the images to PIL format...
    # count_image = Image.fromarray(count_image)

    # ...and then to ImageTk format
    # count_image = ImageTk.PhotoImage(count_image)

    # if the panels are None, initialize them
    # if panelA is None or panelB is None:
    #     # the first panel will store our original image
    #     panelA = Label(image=count_image)
    #     panelA.image = count_image
    #     panelA.pack(side="left", padx=10, pady=10)
    #
    #     # while the second panel will store the edge map
    #     panelB = Label(image=count_image)
    #     panelB.image = count_image
    #     panelB.pack(side="right", padx=10, pady=10)
    #
    # # otherwise, update the image panels
    # else:
    #     # update the pannels
    #     panelA.configure(image=count_image)
    #     panelB.configure(image=count_image)
    #     panelA.image = count_image
    #     panelB.image = count_image

    frame2 = QWidget()  # Replace it with any frame you will putting this label_image on it
    label_Image = QLabel(frame2)
    frame2.setMinimumSize(1000, 500)


    frame2.setWindowTitle('Browser')


    qr = frame2.frameGeometry()
    cp = QDesktopWidget().availableGeometry().center()
    qr.moveCenter(cp)
    frame2.move(qr.topLeft())

    # label_Image.resize(500, 500)
    # frame=np.array(frame).reshape(2048, 2048).astype(np.int32)
    videoFrame = QImage(frame, 1000, 400,
                        QImage.Format_RGB888)
    image_profile = QImage(videoFrame)  # QImage object
    # image_profile = image_profile.scaled(250, 250, aspectRatioMode=Qt.KeepAspectRatio,
    #                                      transformMode=Qt.SmoothTransformation)  # To scale image for example and keep its Aspect Ration
    label_Image.setPixmap(QPixmap(image_profile))
    frame2.show()
    # app.exec_()

    # convertFrame = QtGui.QPixmap(videoFrame)
    # self.imageBox.setPixmap(convertFrame)
    # self.imageBox.show()

    # out.write(frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        # release the video capture and video write objects
        video.release()
        out.release()
        break
