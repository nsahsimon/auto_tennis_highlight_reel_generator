# import cv2
# import numpy as np
# import paddleocr
# import math
# import os
# from PIL import Image, ImageDraw, ImageFont

# # print("Initializing OCR Engine")
# # ocr = paddleocr.PaddleOCR(lang='en')
# # print("Paddle OCR object successfully created")

# def extractScoreBoard(image):
#     # relative coordinates of the top left corner of the scoreboard
#     topLeftRel = (0.017, 0.842) 

#     # relative coordinates of the bottom right corner of the scoreboard
#     bottomRightRel = (0.364, 0.95)

#     # extracting image dimensions
#     width = image.shape[1]
#     height = image.shape[0]

#     # calculating the absolute scoreboard corner coordinates
#     topLeft = (int(width * topLeftRel[0]), int(height * topLeftRel[1]))
#     bottomRight = (int(width * bottomRightRel[0]), int(height * bottomRightRel[1]))


#     # extract the score board 
#     scoreBoard = image[topLeft[1] : bottomRight[1], topLeft[0] : bottomRight[0],:]

#     return scoreBoard


# def decodeResults(results):
#     predictions = results[0]
#     boxes = [prediction[0] for prediction in predictions ]
#     txts = [prediction[1][0] for prediction in predictions]
#     scores = [prediction[1][1] for prediction in predictions]
#     return {'boxes': boxes,'txts': txts, 'scores': scores}


# def draw_boxes(image, boxes, txts, scores, drop_score=0.5, font_path=None):
#     """
#     Draw bounding boxes and text on the input image.
#     """
#     font = ImageFont.truetype(font_path, size=16) if font_path else ImageFont.load_default()
#     for i, box in enumerate(boxes):
#         # Convert score to float if it's a string
#         score = float(scores[i]) if isinstance(scores[i], str) else scores[i]
#         if scores is not None and (score < drop_score or math.isnan(score)):
#             continue
#         tl = box[0]
#         br = box[2]
#         # Draw bounding box
#         cv2.rectangle(image, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), (0,0, 255), 2)
#         # Draw text
#         # text = txts[i][0]
#         # cv2.putText(image, text, (xy[0], xy[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     return image


# def findScoreBoardEdge(image):

#     blur_image = cv2.GaussianBlur(image, (3,3), 0.01)

#     canny_image = cv2.Canny(blur_image, 115, 150)    
    
#     height, width = canny_image.shape

#     # assert(channels == 1)

#     max_dist = 0

#     # Loop over each row and each column of the image
#     for y in range(height):
#         for x in range(width):
#             # Get the color of the pixel at (x, y)
#             # check if the pixel is white
#             if canny_image[y][x] == 255: 
#                 if x > max_dist:
#                     max_dist = x

#     edgeXCoordRel = max_dist / image.shape[1]

#     # print(f"Max distance: {max_dist}")
#     return edgeXCoordRel
  

# def extractFrameData(src, ocr):
#     image = extractScoreBoard(src)
#     original = image
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     width = image.shape[1] ; height = image.shape[0]
#     width_of_dark = int (0.09 * width)
#     edge_x = int (findScoreBoardEdge(image) * width)
#     start_y = 0; end_y = height
#     start_x = edge_x - width_of_dark ; end_x = edge_x
#     image[start_y : end_y,start_x : end_x] = cv2.bitwise_not(image[start_y : end_y,start_x : end_x])
#     image = cv2.bitwise_not(image)
#     image = cv2.equalizeHist(image, )
#     new_start_x = int(edge_x - 0.14 * width)
#     image = original[start_y : end_y,  new_start_x : edge_x]
#     image = cv2.resize(image, (100, 100))
#     results = ocr.ocr(image)
#     results = decodeResults(results)
#     txts = results['txts']
#     print(txts)
#     up = None; down = None
#     if len(txts) == 4:
#         up = [txts[0] , txts[1]]
#         down = [txts[2], txts[3]]
#     return [up, down, edge_x]


# # test function
# def test():
#     file_names = os.listdir(path='images/')
#     for i, name in enumerate(file_names):
#         full_path = os.path.join('images/', name)
#         image = cv2.imread(full_path)
#         data = extractFrameData(image)
#         print(f"{i + 1}: {name}")
#         print(f" up: {data[0]} \n down: {data[1]} \n edge_x: {data[2]} \n")
#         cv2.imshow("image", image)
#         cv2.waitKey()
#         cv2.destroyAllWindows()


# # test()



