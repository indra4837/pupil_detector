import numpy as np
import cv2
import math
import argparse
import time


def calculate_area(contours):
    """ Calculate contour area

    Paramters:
    contours: List[numpy.ndarray]
    
    Returns:
    List[numpy.ndarray]: contours_area
    """
    
    contours_area = []
    # calculate area and filter into new array
    for con in contours:
        area = cv2.contourArea(con)
        if 10000 < area < 60000:
            contours_area.append(con)
    
    return contours_area


def check_circularity(con):
    """ Check circularity of contours and 
    calculate center coords and radius of resulting circle

    Paramters:
    con: numpy.ndarray
    
    Returns:
    float:  circularity
    int:    cX
    int:    cY
    int:    r
    """

    perimeter = cv2.arcLength(con, True)
    area = cv2.contourArea(con)
    if perimeter == 0:
        return 0, 0, 0, 0
    circularity = 4*math.pi*(area/(perimeter*perimeter))

    M = cv2.moments(con)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    r = int(math.sqrt(area/(math.pi)))

    return circularity, cX, cY, r


def detect_contour(gray, img):
    """ Perform Gaussian Blur to smoothen image, binary threshold image to extract features 
    Detects contours on filtered image

    Paramters:
    gray:   numpy.ndarray
    img:    numpy.ndarray
    Returns:
    img:    numpy.ndarray
    """

    filter_img = cv2.GaussianBlur(gray, (11, 11), 11)
    #filter_img = cv2.bilateralFilter(img, 7, 50, 50)
    _, thresh = cv2.threshold(filter_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_area = calculate_area(contours)

    contours_circles = []

    # check if contour is of circular shape
    for con in contours_area:
        circularity, cX, cY, r = check_circularity(con)
        if 0.7 < circularity < 1.1:
            contours_circles.append(con)
            cv2.circle(img, (cX, cY), 3, (0,255,0), 3)
        else:          
            cv2.circle(img, (cX, cY), r, (0,255,0), 3)
            cv2.circle(img, (cX, cY), 3, (0,255,0), 3)

    cv2.drawContours(img, contours_circles, -1, (0, 255, 0), 3)

    return img, filter_img, thresh


def extract_roi(img):
    """ Extract region of interest in frame to perform image processing pipelines

    Paramters:
    img: numpy.ndarray
    
    Returns:
    numpy.ndarray: eye_ROI
    """

    polygons = np.array([(300, 900), (300, 200), (1050, 200), (1050, 900)])
    mask = np.zeros_like(img)
    cv2.fillConvexPoly(mask, polygons, 255)
    eye_ROI = cv2.bitwise_and(img, mask)

    return eye_ROI


def fps_overlay(img, fps):
    """ Overlay FPS onto output img

    Paramters:
    img: numpy.ndarray
    fps: float   

    Returns:
    numpy.ndarray: img
    """

    text = "FPS: {:.2f}".format(fps)

    return cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)


def image_reader(filename, image_layers):
    """ Image reader and performs image processing pipeline on
    image file

    Paramters:
    filename: str
    """

    img = cv2.imread(filename)
    img_cp = img.copy()
    gray = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)
    gray = extract_roi(gray)
    img_cp, filter_img, thresh = detect_contour(gray, img_cp)

    if image_layers:
        cv2.namedWindow("ROI")
        cv2.namedWindow("Gaussian Blur")
        cv2.namedWindow("Thresholding")
        cv2.namedWindow("Output") 
        cv2.imshow("ROI", gray)
        cv2.imshow("Gaussian Blur", filter_img)
        cv2.imshow("Thresholding", thresh)
        cv2.imshow("Output", img_cp)
    else:
        cv2.imshow("Output", img_cp)
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()


def video_reader(filename, image_layers):
    """ Video capture reader and performs image processing pipeline on
    captured frames

    Paramters:
    filename: str
    """
    
    cap = cv2.VideoCapture(filename)
    
    while(True):
        ret, img = cap.read()
        tic = time.time()
        if not ret:
            break
        img_cp = img.copy()
        gray = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)
        gray = extract_roi(gray)
        img_cp, filter_img, thresh = detect_contour(gray, img_cp)
        toc = time.time()
        fps = 1/(toc-tic)
        img_cp = fps_overlay(img_cp, fps)

        if image_layers:
            cv2.namedWindow("ROI")
            cv2.namedWindow("Gaussian Blur")
            cv2.namedWindow("Thresholding")
            cv2.namedWindow("Output") 
            cv2.imshow("ROI", gray)
            cv2.imshow("Gaussian Blur", filter_img)
            cv2.imshow("Thresholding", thresh)
            cv2.imshow("Output", img_cp)
        else:
            cv2.imshow("Output", img_cp)
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            break
        if key == ord('p'):
            cv2.waitKey(-1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_format',
                        type=int,
                        dest='input_format',
                        default=1,
                        help='Image(0) or Video(1)')
    parser.add_argument('--input_file',
                        type=str,
                        dest='input_file',
                        default='/home/indra/Documents/Telemedc/pupil_detector/assets/sample.mkv',
                        help='Path to input file (image or video)')
    parser.add_argument('--image_layers',
                        type=bool,
                        dest='image_layers',
                        default=False,
                        help='Open CV Windows to see intermediate processing')
    
    args = parser.parse_args()

    if args.input_format:
        video_reader(args.input_file, args.image_layers)
    else:
        image_reader(args.input_file, args.image_layers)