from darkflow.net.build import TFNet
import cv2
import sys

options = {"model": "cfg/yolo-obj.cfg", "load": "bin/yolo-obj_300.weights", "threshold": 0.6}

tfnet = TFNet(options)

# Read video
video = cv2.VideoCapture("./sample_img/test32.mp4")
#imgcv = cv2.imread("./sample_img/IMG_0001.JPEG")

# Set up tracker.
tracker = cv2.TrackerKCF_create()
tracker2 = cv2.TrackerKCF_create()

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

previous_x = [] #stores the
previous_y = []
previous_id = []
yield_object_total_count = []

# Define an initial bounding box
#bbox = (625, 252, 123, 135)

# Uncomment the line below to select a different bounding box
#bbox = cv2.selectROI(frame, False)
# bbox2 = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
#ok = tracker.init(frame, bbox)
# ok = tracker2.init(frame, bbox2)

initTrackBbox=0
while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    result = tfnet.return_predict(frame)
    print(result)
    for one_detection in result:
        left = one_detection['topleft']['x']
        top = one_detection['topleft']['y']
        right = one_detection['bottomright']['x']
        bot = one_detection['bottomright']['y']
        cv2.rectangle(frame, (left, top), (right, bot), (0, 255, 0), 2, 1)

        if initTrackBbox==0:
            #bbox = (625, 252, 123, 135)
            bbox = (left, top, right-left, bot-top)
            for value in bbox:
                print(value)
                print('----')
            #ok = tracker.init(frame, bbox)
            initTrackBbox = 1
        if initTrackBbox==2:
            #bbox = (625, 252, 123, 135)
            bbox2 = (left, top, right-left, bot-top)
            for value in bbox2:
                print(value)
                print('----')
            #ok = tracker.init(frame, bbox)
            initTrackBbox = 3

    #for 1 athule tiyena varible elyatath valid
    if initTrackBbox == 1:
        # bbox = (625, 252, 123, 135)
        #bbox = (left, top, right - left, bot - top)
        for value in bbox:
            print(value)
            print('----')
        ok = tracker.init(frame, bbox)
        initTrackBbox = 2

    if initTrackBbox == 3:
        # bbox = (625, 252, 123, 135)
        #bbox = (left, top, right - left, bot - top)
        for value in bbox2:
            print(value)
            print('----')
        ok = tracker2.init(frame, bbox2)
        initTrackBbox = 4

    for value in bbox:
        print(value)
        print('----')

    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)
    ok, bbox2 = tracker2.update(frame)
    # ok, bbox2 = tracker2.update(frame)
    # for value in bbox:
    #     print(value)
    # print('----')

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        p3 = (int(bbox2[0]), int(bbox2[1]))
        p4 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
        cv2.rectangle(frame, p3, p4, (255, 0, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display tracker type on frame
    #cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27: break





