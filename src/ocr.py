import cv2
import pytesseract
import numpy as np
import keras_ocr
import time

# Load the Keras-OCR model
pipeline = keras_ocr.pipeline.Pipeline()

def kerasocrPredict(images, isSingleChannel=True):

    if isSingleChannel == True:
        images = [cv2.merge((image, image, image)) for image in images]

    predictions = pipeline.recognize(images)

    return ["".join([prediction[0] for prediction in predictions[i]]) for i in range(len(predictions)) ]


def detectText(images):
    results = {'tl':[], 'tr':[], 'bl':[], 'br':[]}
    edges = []
    for section in ('tl', 'tr', 'bl', 'br'):
        bin_iois = []
        for image in images:
            # cv2.imshow("scr_brd", image)
            # cv2.waitKey(5)

            # resize image
            resizedWidth = 667
            resizedHeight = 150
            resized_image = cv2.resize(image, (resizedWidth, resizedHeight))
             
            # convert the image into grayscale
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

            # Perform adaptive thresholding
            thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            blur_image = cv2.GaussianBlur(gray_image, (3,3), 0.01)

            canny_image = cv2.Canny(blur_image, 115, 150)

            # Find the X-coordinate of the edge
            edgeXCoord = int (findScoreBoardEdge2(canny_image) * resizedWidth)

            # define the region of interest
            roiWidth  = int (0.155 * resizedWidth)

            # All
            # roi = (((edgeXCoord - roiWidth),0 ),(edgeXCoord,resizedHeight))

            # Top
            # roi = (((edgeXCoord - roiWidth),0 ),(edgeXCoord,resizedHeight))

            # Top Left
            if section == 'tl':
                roi = (((edgeXCoord - roiWidth),0 ),(edgeXCoord - roiWidth // 2,resizedHeight // 2)) # (tl(width, height), br(width, height))
            
            #Top Right
            if section == 'tr':
                roi = (((edgeXCoord  - roiWidth // 2),0 ),(edgeXCoord ,resizedHeight // 2))
            
            # Bottom Left
            if section == 'bl':
                roi = (((edgeXCoord - roiWidth), resizedHeight // 2 ),(edgeXCoord - roiWidth // 2,resizedHeight)) # (tl(width, height), br(width, height))

            # Bottom Right
            if section == 'br':
                roi = (((edgeXCoord  - roiWidth // 2),resizedHeight // 2 ),(edgeXCoord ,resizedHeight))
            
            # cv2.line(resized_image, (edgeXCoord, 0), (edgeXCoord, resizedHeight),(255, 255, 0), 3)

            # extract image of interest
            # ioi = resized_image[roi[0][1]: roi[1][1], roi[0][0]: roi[1][0]]
            ioi = gray_image[roi[0][1]: roi[1][1], roi[0][0]: roi[1][0]]

            # if section == 'tl' or section == 'bl':
            #     _ , bin_ioi = cv2.threshold(ioi, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # bin_ioi = cv2.adaptiveThreshold(ioi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            bin_ioi = None

            if section == 'tr' or section == 'br':
                bin_ioi = ioi

            if section == 'tl' or section == 'bl':
                bin_ioi = cv2.bitwise_not(ioi) 

            # This is done only once
            if section == 'tr':
                edges.append(edgeXCoord)

            if section == 'bl':
                cv2.imshow("img", bin_ioi)
                cv2.waitKey()

            cv2.destroyAllWindows()

            bin_iois.append(bin_ioi)


        # img = bin_ioi[0 : int (bin_ioi.shape[0] / 2), 0 : int (bin_ioi.shape[1] / 2)]
        # cv2.imshow("img", img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        # result = pytesseract.image_to_string(img)
        keras_bin_images = []
        batchSize = 20

        # organize the binary images in groups of "batch size" or less
        keras_bin_images = [bin_iois[idx - batchSize : idx ] for idx in range(1, 1 + 2 * int(len(bin_iois) / batchSize)) if idx % batchSize == 0] + [bin_iois[-(len(bin_iois) % batchSize) : ]]
        

        time1 = time.time_ns() 
        for image_batch in keras_bin_images:
            results[section] += kerasocrPredict(image_batch)
        inference_time = ( time.time_ns() - time1) // 1000000000
        
        print(f"It took {inference_time} seconds for {len(bin_iois)} images")
        print(f"Section {section}: {results[section]}")

    cv2.destroyAllWindows()
    
    return [(results['tl'][i], results['tr'][i], results['bl'][i], results['br'][i], edges[i]) for i in range(len(results['tl']))]


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
    bottomRightRel = (0.364, 0.95)

    # extracting image dimensions
    width = image.shape[1]
    height = image.shape[0]

    # calculating the absolute scoreboard corner coordinates
    topLeft = (int(width * topLeftRel[0]), int(height * topLeftRel[1]))
    bottomRight = (int(width * bottomRightRel[0]), int(height * bottomRightRel[1]))


    # extract the score board 
    scoreBoard = image[topLeft[1] : bottomRight[1], topLeft[0] : bottomRight[0],:]

    return scoreBoard


# Method 1 using white pixel centroid
def findScoreBoardEdge1(image):

    moments = cv2.moments(image)

    # Calculate centroid of white pixels
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])

    edgeXCoordRel = 2 * cX / image.shape[1]

    # print(f"Max distance: {cX}")

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

    # print(f"Max distance: {max_dist}")
    return edgeXCoordRel


images = []
paths_idx = [i for i in range(15)]

images = [cv2.imread(f"images/{idx + 1}.png") for idx in range(20)]
scoreboards  =  [extractScoreBoard(image) for image in images]
detectedTexts = detectText(scoreboards)
print(detectedTexts[1])
# cv2.imshow("Sample image", image)
# cv2.waitKey()
# cv2.destroyAllWindows()


# scoreboard = extractScoreBoard(image)
# cv2.imshow('score board', scoreboard)
# cv2.waitKey()
# cv2.destroyAllWindows()


# detectText(scoreboard)