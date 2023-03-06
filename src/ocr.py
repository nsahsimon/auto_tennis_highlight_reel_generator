import cv2
import pytesseract
import numpy as np


def detectText(image):
    # convert the image into grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # resize image
    resizedWidth = 450
    resizedHeight = 100
    resized_image = cv2.resize(gray_image, (resizedWidth, resizedHeight))
     
    # Perform adaptive thresholding
    thresh_image = cv2.adaptiveThreshold(resized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    blur_image = cv2.GaussianBlur(resized_image, (3,3), 0.01)
    canny_image = cv2.Canny(blur_image, 115, 150)

    erodeKernel = np.ones((3,3), np.uint8)

    eroded_thresh_image = cv2.dilate(thresh_image, erodeKernel, iterations=1)
    right_canny_image = cv2.bitwise_and(canny_image, cv2.bitwise_not(eroded_thresh_image), mask=None)

    # Define the kernel for dilation
    kernel = np.ones((5,5), np.uint8)

    # Perform dilation
    dilated_image = cv2.dilate(canny_image, kernel, iterations=1)
    final_image = cv2.bitwise_and(thresh_image, dilated_image, mask=None)
    final_image_inv = cv2.bitwise_and(cv2.bitwise_not(thresh_image), dilated_image, mask=None)
    thresh_image_inv = cv2.bitwise_not(thresh_image)
    some_image = cv2.bitwise_and(final_image_inv, thresh_image_inv, mask=None)

    # Find the X-coordinate of the edge
    edgeXCoord = int (findScoreBoardEdge2(canny_image) * resized_image.shape[1])

    # Draw the edge line
    cv2.line(resized_image, (edgeXCoord, 0), (edgeXCoord, resized_image.shape[0]), (255, 255, 0))


    # get the xcoordinate of the line dividing the points from sets
    widthOfPointsSection = 30
    whiteBandWidth = 10
    padding = 10
    dividerLineXCoord = edgeXCoord - widthOfPointsSection

    # remove white band
    try:
        final_image[:, (dividerLineXCoord - whiteBandWidth) : dividerLineXCoord] = np.zeros((resizedHeight, whiteBandWidth),  dtype=np.uint8)
    except:
        print("Unable to remove white band 1")
    try:
        final_image_inv[:, (edgeXCoord - 3) : edgeXCoord + 5] = np.zeros((resizedHeight,8),  dtype=np.uint8)
    except:
        print("unable to reove white band 2")


    section1 = final_image[:,:dividerLineXCoord]
    section2 = final_image_inv[:,dividerLineXCoord:]

    ultimateImage = np.zeros((resizedHeight, resizedWidth), dtype=np.uint8)
    ultimateImage[:,:dividerLineXCoord] = section1
    ultimateImage[:,dividerLineXCoord:] = section2 

    # pad section2
    tempSection = np.zeros((section2.shape[0], section2.shape[1] + padding), dtype=np.uint8)
    tempSection[:,padding:] = section2
    section2 = tempSection



    upperHalf = cv2.bitwise_not(ultimateImage[:int (ultimateImage.shape[0] / 2),:])
    lowerHalf = cv2.bitwise_not(ultimateImage[int (ultimateImage.shape[0] / 2) : ,:])

    upperHalf1 = cv2.bitwise_not(section1[:int (section1.shape[0] / 2),:])
    lowerHalf1 = cv2.bitwise_not(section1[int (section1.shape[0] / 2) : ,:])

    upperHalf2 = cv2.bitwise_not(section2[:int (section2.shape[0] / 2),:widthOfPointsSection * 2])
    lowerHalf2 = cv2.bitwise_not(section2[int (section2.shape[0] / 2) : ,:widthOfPointsSection * 2])
    
    # upperHalf = resized_image[:int (resized_image.shape[0] / 2),:]
    # lowerHalf = resized_image[int (resized_image.shape[0] / 2) : ,:]

    # figs, axes = plt.subplots(2,3)

    # axes[0][0].set_title("utlimate text")
    # axes[0][0].imshow(ultimateImage)
    # axes[0][1].set_title("section 1")
    # axes[0][1].imshow(section1)
    # axes[0][2].set_title("section 2")
    # axes[0][2].imshow(section2)
    # axes[1][0].set_title("upper half 2")
    # axes[1][0].imshow(upperHalf2)

    # plt.show()

    
    upperText = [pytesseract.image_to_string(section) for section in [upperHalf1, upperHalf]]
    lowerText = [pytesseract.image_to_string(section) for section in [lowerHalf1, lowerHalf]]

    # upperText = kerasocrPredict([upperHalf1, upperHalf])
    # lowerText = kerasocrPredict([lowerHalf1, lowerHalf])

    print(f"upper raw text: {repr(upperText)}, \nlower raw text: {repr(lowerText)}")

    upperSetDigits = [s for s in "".join(str.split(upperText[0])) if s.isdigit()] 
    lowerSetDigits = [s for s in "".join(str.split(lowerText[0])) if s.isdigit()] 
    print(f"upper: {upperSetDigits} \nlower: {lowerSetDigits}")

    upperPointDigits = [s for s in "".join(str.split(upperText[1])) if s.isdigit()] 
    lowerPointDigits = [s for s in "".join(str.split(lowerText[1])) if s.isdigit()] 

    for d in upperSetDigits:
        if d in upperPointDigits:
            upperPointDigits.remove(d)

    for d in lowerSetDigits:
        if d in lowerPointDigits:
            lowerPointDigits.remove(d)

    print(f"upper Points: {''.join(upperPointDigits)} \nlower Points: {''.join(lowerPointDigits)}")

    up = ["".join(upperSetDigits), ''.join(upperPointDigits)]
    down = ["".join(lowerSetDigits), ''.join(lowerPointDigits)]


    print(f"up: {up} \ndown: {down}")


    # reader = easyocr.Reader(['en'])
    # results = reader.readtext(lowerHalf)
    # for result in results:
    #     print(result[1])

    return [up, down, edgeXCoord] # will use edgetXCoord to detect changes in sets



def findScoreBoardEdge1(image):

    moments = cv2.moments(image)

    # Calculate centroid of white pixels
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])

    edgeXCoordRel = 2 * cX / image.shape[1]

    print(f"Max distance: {cX}")

    return edgeXCoordRel


# Method 2 using minArea bounding rect
def findScoreBoardEdge2(canny_image):

    # Method 2 using min bounding rect

    height, width = canny_image.shape

    # assert(channels == 1)

    max_dist = 0
    # Loop over each row and each column of the image
    for y in range(height):
        for x in range(width):
            # Get the color of the pixel at (x, y)
            # check if the pixel is white
            if canny_image[y][x] == 255: 
                if x > max_dist:
                    max_dist = x

    edgeXCoordRel = max_dist / canny_image.shape[1]
    print(f"Max distance: {max_dist}")
    return edgeXCoordRel


# Takes the path to the video and processes the various frames
# sampleTime the length of the video in seconds to be processed
# samplePeriod is the interval of time after which new samples are taken

def extractFrameData(frame):
    scoreBoard = extractScoreBoard(frame)
    return detectText(scoreBoard)


def extractScoreBoard(image):
    # relative coordinates of the top left corner of the scoreboard
    topLeftRel = (0.017, 0.842) 

    # relative coordinates of the bottom right corner of the scoreboard
    bottomRightRel = (0.364, 0.958)

    # extracting image dimensions
    width = image.shape[1]
    height = image.shape[0]

    # calculating the absolute scoreboard corner coordinates
    topLeft = (int(width * topLeftRel[0]), int(height * topLeftRel[1]))
    bottomRight = (int(width * bottomRightRel[0]), int(height * bottomRightRel[1]))


    # extract the score board 
    scoreBoard = image[topLeft[1] : bottomRight[1], topLeft[0] : bottomRight[0],:]

    # cv2.imshow("Extracted score board", scoreBoard)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return scoreBoard
