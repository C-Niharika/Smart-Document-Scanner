'''
https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
'''
import cv2 as cv
import numpy as np

def detect_boundary(image,clow=50,chigh=100,show_contours = False):
    '''
    to detect boundary
    input: rgb image
    steps:
        convert to garyscale
        apply gaussian blur
        find canny edge thresholds using otsu's method
        apply erosin dilation to join broken edges
        find contours and select the one with largest area
        find convex hull of this contour
        find lower polynomial approximation of this contour
    output: lower polynomial approximated contour
    '''
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    image = cv.GaussianBlur(image,(5,5),0)
    # find thresh_value from otsu
    val,thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);

    thresh = cv.Canny(image, val/2, val, None, 3)
    # cv.imwrite('canny.jpg',thresh)
    # run erosion dilation to join broken edges
    kernel = np.ones((21,21),np.uint8)
    thresh = cv.dilate(thresh,kernel,iterations = 1)
    thresh = cv.erode(thresh,kernel,iterations = 1)

    # sort contours by area and take the largest one
    contours,hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv.contourArea,reverse=True)
    if show_contours:
        thresh_color = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
        colors = [(128,128,128),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
        # thresh_color = bgr
        for c in range(min(4,len(contours))):
            color = colors[c%len(colors)]
            cv.drawContours(thresh_color, contours, c, color, 3)

    # find the convex hull of the contour
    cnt = contours[0]
    cnt = cv.convexHull(cnt)

    # find lower order approximation
    epsilon = 0.1 * cv.arcLength(cnt,True)
    cnt_ = cv.approxPolyDP(cnt,epsilon,True)
    # sometimes this approximation becomes line. so if area  is zero we don't use the approximated contour
    if cv.contourArea(cnt_)>0:
        cnt_ = np.vstack(cnt_).squeeze()
        if len(cnt_) == 4:
            return cnt_
        else:
            print(cnt_)
    # if area is zero go for minrect bounding the contour as boundaries.
    # cv.drawContours(thresh_color,[cnt],0,(255,255,0),4)
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    # print(rect,'\n',box)
    # cv.drawContours(thresh_color,[box],0,(255,0,0),4)
    if show_contours:
        h,w,c = thresh_color.shape
        scale = 0.5*min(1920/w,1080/h)
        thresh_color = cv.resize(thresh_color,(int(scale*w),int(scale*h)))
        cv.imshow('debug',thresh_color)
    return box

def crop_and_transform(original_image,boundary):
    '''
    input: original_image (morphed/distorted image)
         and boundary
    output: rectified image
    steps:
        order the corners of the boundary in top-left, top-right, bottom-right and bottom-left order
        this is important otherwise rectified image will be rotated

        find the largest width and heights and consider it as the width and height of the real document
        find perspective tranform matrix
        rectify the distorted image with this matrix
    '''
    boundary = np.array(boundary)
    boundary = boundary.reshape(4,2)
    morphed_rect = np.zeros((4, 2),dtype =np.float32)

    # Top left corner should contains the smallest sum,
    # Bottom right corner should contains the largest sum
    s = np.sum(boundary, axis=1)
    morphed_rect[0] = boundary[np.argmin(s)]
    morphed_rect[2] = boundary[np.argmax(s)]

    # top-right will have smallest difference
    # botton left will have largest difference
    diff = np.diff(boundary, axis=1)
    morphed_rect[1] = boundary[np.argmin(diff)]
    morphed_rect[3] = boundary[np.argmax(diff)]

    # now find the maximum height and width in the image and consider it
    # h and w of original image
    (tl, tr, br, bl) = morphed_rect
    w1 =  np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
    w2 = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
    w = max(w1,w2)

    h1 = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
    h2 = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
    h = max(h1,h2)

    w,h = int(w),int(h)

    # cordinates of the rectangular image
    original_rect = np.array([
            [0,0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]], dtype="float32")

    M = cv.getPerspectiveTransform(morphed_rect, original_rect)
    new_image = cv.warpPerspective(original_image, M, (w, h))
    return new_image


#########################################################################

def test_crop_and_transform(filename=None):
    if filename is None:
        filename = 'test.jpg'
    img = cv.imread(filename)
    boundary= np.array([[116,494], [111,172], [353,168], [358, 490]])
    crop = crop_and_transform(img,boundary)
    cv.imshow('crop',crop)
    cv.waitKey()

def test_detect_boundary(image_path=None):
    if not image_path:
        image_path = 'test.jpg'
    image = cv.imread(image_path)
    rect = detect_boundary(image,show_contours=True)
    cv.drawContours(image,[rect],0,(255,0,0),4)
    # r = np.array([[1396,2624], [1351,  619], [3934,  560], [3979, 2565]])
    # cv.drawContours(image,[r],0,(255,255,255),4)

    h,w,c = image.shape
    scale = 0.5*min(1920/w,1080/h)
    image = cv.resize(image,(int(scale*w),int(scale*h)))
    cv.imshow('rect',image)
    cv.waitKey()


if __name__ == '__main__':
    filename = 'data/test.jpg'
    # filename = 'data/line25.jpg'
    test_detect_boundary(filename)
    # test_crop_and_transform()
