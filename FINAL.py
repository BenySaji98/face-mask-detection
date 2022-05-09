from sanitizer import drop_solution
from tempature import scan_temp_and_display
from mask_detector import load_configs, find_mask

from sanitizer import cleaup as s_cleanup
from tempature import cleaup as t_cleanup
from mask_detector import cleaup as ai_cleanup
import argparse


# command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,default="face_detector",
        help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,default="mask_detector.model",
        help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
maskNetPath = args["model"]

load_configs(prototxtPath, weightsPath, maskNetPath)

try:
    print("use CTRL-C for Keyboard Interrupt")

    while True:
        yes_mask, no_mask, frame = find_mask(args['confidence'],True)

        # show the output frame
        cv2.imshow("Face Mask Detector", frame)

        total_faces = yes_mask + no_mask
        print(f"found {total_faces} faces with {yes_mask} mask")

        if total_faces == 0: # skip
            continue
        if total_faces > 1:
            print("Warning! Only one person allowed!")
            time.sleep(5)
            continue

        if yes_mask == 1:
            print("Found mask")
            temperature = scan_temp_and_display()
            drop_solution()
        else:
            print("No mask found")
except KeyboardInterrupt:
    print("Received KeyboardInterrupt")
finally:
    s_cleanup()
    t_cleanup()
    ai_cleanup()
