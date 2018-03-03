from darkflow.net.build import TFNet
import cv2
import sys

options = {"model": "cfg/yolo-obj.cfg", "load": "bin/yolo-obj_300.weights", "threshold": 0.6}

tfnet = TFNet(options)

# Set up tracker.
tracker = cv2.MultiTracker_create()
init_once = False

# Read video
video = cv2.VideoCapture("./sample_img/test32.mp4")

# Exit if video not opened.
if not video.isOpened():
    print('Could not open video')
    sys.exit()

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
frame_count=0

while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    same_detections_this_frame = []
    bounding_boxes_for_tracker = []
    current_frame_detection_id = 0
    result = tfnet.return_predict(frame)
    print(result)
    for one_detection in result:
        left = one_detection['topleft']['x']
        top = one_detection['topleft']['y']
        right = one_detection['bottomright']['x']
        bot = one_detection['bottomright']['y']
        mid_x = left + (right - left) / 2
        mid_y = top + (bot - top) / 2
        cv2.rectangle(frame, (left, top), (right, bot), (0, 255, 0), 2, 1)
        if mid_y > 100 and mid_y<600:  # safe upper margin
            # current_frame_detection_id +=1
            same_detection_has_found = 0
            print("-----current_frame_detection_id ------", current_frame_detection_id)

            if is_first_frame == 1:
                new_detection = 1
                yield_object_total_count += 1
            if is_first_frame == 0:
                for index in range(len(tracking_list)):
                    # print('---11111111111-------')
                    print("----(tracking_list)index----", index)
                    detect_and_track_x_diff = abs(mid_x - tracking_list[index][0])
                    detect_and_track_y_diff = abs(mid_y - tracking_list[index][1])
                    print(detect_and_track_x_diff)
                    print(detect_and_track_y_diff)

                    if same_detection_has_found == 0:
                        if detect_and_track_x_diff < 16 and detect_and_track_y_diff < 29:
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
            del bounding_boxes_for_tracker[index1] # removing previous detected boxes
    print("len(bounding_boxes_for_tracker) after", len(bounding_boxes_for_tracker))

    if len(bounding_boxes_for_tracker) > 0:  # new bboxes to add
        new_detection = 1
        yield_object_total_count += 1

    if ok:
        if not init_once:
            if new_detection == 1:
                for index in range(len(bounding_boxes_for_tracker)):
                    ok = tracker.add(cv2.TrackerMIL_create(), frame, bounding_boxes_for_tracker[index])
                # ok = tracker.add(cv2.TrackerMIL_create(), frame, bbox2)
                # init_once = True
                new_detection = 0

        # Update tracker
        ok, boxes = tracker.update(frame)
        print(ok, boxes)

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

    # Start timer
    # timer = cv2.getTickCount()

    # Calculate Frames per second (FPS)
    # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Display FPS on frame
    frame_count+= 1
    cv2.putText(frame, "Frame No : " + str(int(frame_count)), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    # cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
    cv2.putText(frame, "Count : " + str(int(yield_object_total_count)), (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2);
    # medium size tomato weight 123g

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27: break
