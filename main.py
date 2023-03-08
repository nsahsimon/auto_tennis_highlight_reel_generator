import tkinter as tk
from tkinter import filedialog
import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, 'src')
import moviepy.editor as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
import paddleocr
from PIL import Image, ImageDraw, ImageFont

SET_CHANGE_THRESH = 10
POINT_TIME_OFFSET = 15
ODD_GAME_TIME_OFFSET = 70
DEFAULT_SAMPLE_TIME = 1200
DEFAULT_SAMPLE_PERIOD = 8


# Frame class
class Frame:
    count = None # the position of the frame in the list of all frames in the video
    timestamp = None # the timestamp of the frame given the current frame rate
    fps = None # the frame rate of the video
    data = [None, None, None] # a tuple containing the game score contained in the frame (upDigits, downDigits, edgeXCoord)
    image = None
    
    def __init__(self, count : np.random.randint(1, 100) = 10, timestamp: int = np.random.randint(1, 100), fps : float= 0.1 * np.random.randint(1, 100), data: list = [], image: np.ndarray = np.zeros((3,3),dtype=np.uint8)):
        self.count = count
        self.timestamp = timestamp
        self.fps = fps
        self.data = data
        self.image = image

    def classicTimestamp(self):
        return time.strftime("%H:%M:%S", time.gmtime(self.timestamp))
    
    def setEdge_x(self, edge_x):
        self.data[2] = edge_x


# IMPORTANT OCR FUNCTIONS

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


def decodeResults(results):
    predictions = results[0]
    boxes = [prediction[0] for prediction in predictions ]
    txts = [prediction[1][0] for prediction in predictions]
    scores = [prediction[1][1] for prediction in predictions]
    return {'boxes': boxes,'txts': txts, 'scores': scores}


def draw_boxes(image, boxes, txts, scores, drop_score=0.5, font_path=None):
    """
    Draw bounding boxes and text on the input image.
    """
    font = ImageFont.truetype(font_path, size=16) if font_path else ImageFont.load_default()
    for i, box in enumerate(boxes):
        # Convert score to float if it's a string
        score = float(scores[i]) if isinstance(scores[i], str) else scores[i]
        if scores is not None and (score < drop_score or math.isnan(score)):
            continue
        tl = box[0]
        br = box[2]
        # Draw bounding box
        cv2.rectangle(image, (int(tl[0]), int(tl[1])), (int(br[0]), int(br[1])), (0,0, 255), 2)
        # Draw text
        # text = txts[i][0]
        # cv2.putText(image, text, (xy[0], xy[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return image


def findScoreBoardEdge(image):

    blur_image = cv2.GaussianBlur(image, (3,3), 0.01)

    canny_image = cv2.Canny(blur_image, 115, 150)    
    
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

    edgeXCoordRel = max_dist / image.shape[1]

    # print(f"Max distance: {max_dist}")
    return edgeXCoordRel


def extractFrameData(src, ocr):
    image = extractScoreBoard(src)
    original = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    width = image.shape[1] ; height = image.shape[0]
    width_of_dark = int (0.09 * width)
    edge_x = int (findScoreBoardEdge(image) * width)
    start_y = 0; end_y = height
    start_x = edge_x - width_of_dark ; end_x = edge_x
    image[start_y : end_y,start_x : end_x] = cv2.bitwise_not(image[start_y : end_y,start_x : end_x])
    image = cv2.bitwise_not(image)
    image = cv2.equalizeHist(image, )
    new_start_x = int(edge_x - 0.14 * width)
    image = original[start_y : end_y,  new_start_x : edge_x]
    image = cv2.resize(image, (100, 100))
    results = ocr.ocr(image)
    results = decodeResults(results)
    txts = results['txts']
    print(txts)
    up = None; down = None
    if len(txts) == 4:
        up = [txts[0] , txts[1]]
        down = [txts[2], txts[3]]
    return [up, down, edge_x]


# test function
def test(self):
    file_names = os.listdir(path='images/')
    for i, name in enumerate(file_names):
        full_path = os.path.join('images/', name)
        image = cv2.imread(full_path)
        data = self.extractFrameData(image)
        print(f"{i + 1}: {name}")
        print(f" up: {data[0]} \n down: {data[1]} \n edge_x: {data[2]} \n")
        cv2.imshow("image", image)
        cv2.waitKey()
        cv2.destroyAllWindows()





# Load paddle ocr
OCR = paddleocr.PaddleOCR(lang='en')


def select_video_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.mov;*.avi;*.mkv")], title="Select a video")
    root.destroy()
    return file_path

def select_directory():
    # Create a tkinter window
    root = tk.Tk()
    # Hide the window
    root.withdraw()
    # Ask user to select a directory
    directory_path = filedialog.askdirectory()

    # Destroy Window
    root.destroy()

    # Return the selected directory path
    return directory_path

def getSampleFrames(path, numberOfSamples=5, samplePeriod=5): # sample period defines how often video frames should be sampled.
    cam = cv2.VideoCapture(path)

    if not cam.isOpened():
        print("Error opening video file")
        exit()
    
    fps = cam.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second: {fps}")

    sampleCount = 1
    while True:
        ret, frame = cam.read()

        if not ret:
            return
        
        if frameCount % int(fps * samplePeriod) == 0:
            cv2.imwrite(f"images/{sampleCount}.png", frame)
            cv2.imshow(f"sample: {sampleCount}",frame)
            cv2.waitKey(10)
            sampleCount += 1
        
        if sampleCount > numberOfSamples:
            break
        frameCount += 1
    return

def processFrames(path: str, sampleTime: int = DEFAULT_SAMPLE_TIME, samplePeriod:int=15):
        sampledFrames = []

        cam = cv2.VideoCapture(path)
        if not cam.isOpened():
            print("Unable to open video")
            exit()

        fps = cam.get(cv2.CAP_PROP_FPS)
        totalFrameCount = cam.get(cv2.CAP_PROP_FRAME_COUNT)
        numberOfSamples = int(sampleTime / samplePeriod)

        print(f"Frames per second: {fps}")
        print(f"Total frame count: {totalFrameCount}")

        frameCount = 0
        sampleCount = 0

        print("Started processing sample frames")

        # holds the data of the previous frame
        # initialized to zero everywhere
        prevFrameData = [['0', '0'], ['0', '0'], 0]

        while True:
            ret, frame = cam.read()
            if not ret:
                print("Could not retrieve video frame")
                break
            
            if frameCount % int(fps * samplePeriod) == 0:
                try:
                    cv2.destroyAllWindows()
                except:
                    pass

                # cv2.imshow(f"Sample: {sampleCount} of {numberOfSamples}",frame)
                # cv2.waitKey(10)

                newFrameData = extractFrameData(frame, ocr=OCR)
                

                # ignore corrupt data
                if newFrameData[0] is None:
                    newFrameData = prevFrameData
                else:
                    # update previous frame data
                    prevFrameData = newFrameData


                print(f"edge_x: {newFrameData[2]}")

                newFrame = Frame(
                    count=frameCount,
                    timestamp= calcTimestamp(fps, frameCount),
                    fps=fps,
                    data=newFrameData,
                    image = frame
                )

                sampledFrames.append(newFrame)
                sampleCount += 1
        
            if sampleCount >= numberOfSamples:
                break

            frameCount += 1

        print(f"Video successfully processed!!")

        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        return sampledFrames

def hasChanged_set(prevFrame: Frame, currentFrame: Frame):
    prevEdgeXCoord = prevFrame.data[2]
    currentEdgeXCoord = currentFrame.data[2]
    if abs(prevEdgeXCoord - currentEdgeXCoord) >= SET_CHANGE_THRESH:
        print(f"Has changed set from: {prevEdgeXCoord} to {currentEdgeXCoord}")
        return True
    else:
        return False

def hasChanged_game(prevFrame: Frame, currentFrame: Frame):
    prevTopleftDigits = prevFrame.data[0][0]
    prevBottomleftDigits = prevFrame.data[1][0]
    currentTopleftDigits = currentFrame.data[0][0]
    currentBottomleftDigits = currentFrame.data[1][0]
    if prevTopleftDigits != currentTopleftDigits or prevBottomleftDigits != currentBottomleftDigits:
        return True
    else:
        return False

def hasChanged_point(prevFrame: Frame, currentFrame: Frame):
    prevToprightDigits = prevFrame.data[0][1]
    prevBottomrightDigits = prevFrame.data[1][1]
    currentToprightDigits = currentFrame.data[0][1]
    currentBottomrightDigits = currentFrame.data[1][1]
    if prevToprightDigits != currentToprightDigits or prevBottomrightDigits != currentBottomrightDigits:
        return True
    else:
        return False
    
# This function splits and organizes the sample frames into SETS -> GAMES -> POINTS
# it returns an organized list of match data
def detectChangesAndSplitFrames(sampledFrames):
    sets = [[[[sampledFrames[0], None], ] ] ] # [set[game[point[start_frame, stop_frame]]]
    for i in range(1, len(sampledFrames)):
        if hasChanged_set(sampledFrames[i - 1], sampledFrames[i]):

            # ADD THE CLOSING FRAME OF THE LASTEST POINT
            latestSet = sets[len(sets) - 1]
            latestGame = latestSet[len(latestSet) - 1]
            latestPoint = latestGame[len(latestGame) - 1]
            latestPoint[1] = sampledFrames[i] # Set the end frame for the latest form
            latestGame[len(latestGame) - 1] = latestPoint
            latestSet[len(latestSet) - 1] = latestGame
            sets[len(sets) - 1] = latestSet

            # ADD a new set
            latestSet = [[[sampledFrames[i], None]]]
            sets.append(latestSet)
            
            continue

        if hasChanged_game(sampledFrames[i - 1], sampledFrames[i]):

            # ADD THE CLOSING FRAME OF THE LAST POINT
            latestSet = sets[len(sets) - 1]
            latestGame = latestSet[len(latestSet) - 1]
            latestPoint = latestGame[len(latestGame) - 1]
            latestPoint[1] = sampledFrames[i] # Set the END frame for the latest form
            latestGame[len(latestGame) - 1] = latestPoint
            latestSet[len(latestSet) - 1] = latestGame
            sets[len(sets) - 1] = latestSet

            # ADD THE NEW GAME
            latestGame = [[sampledFrames[i], None]]
            latestSet.append(latestGame)

            # MODIFY THE LAST SET
            sets[len(sets) - 1] = latestSet
           
            continue

        if hasChanged_point(sampledFrames[i - 1], sampledFrames[i]):
            # ADD THE CLOSING FREAME OF THE LAST POINT
            latestSet = sets[len(sets) - 1]
            latestGame = latestSet[len(latestSet) - 1]
            latestPoint = latestGame[len(latestGame) - 1]
            latestPoint[1] = sampledFrames[i] # Set the END frame for the latest form
            latestGame[len(latestGame) - 1] = latestPoint
            latestSet[len(latestSet) - 1] = latestGame
            sets[len(sets) - 1] = latestSet

            # ADD THE NEW POINT
            latestPoint = [sampledFrames[i], None]
            latestGame.append(latestPoint)

            # Modify the LATEST GAME
            latestSet[len(latestSet) - 1] = latestGame

            # MODIFY THE LAST SET
            sets[len(sets) - 1] = latestSet

            continue
    
    return sets


# This function allows the user to select the clips which he will like to include in the selected reel
# it takes in the organized set of Key point
# returns a list of the indices of the selected key points or clips
# output format :[(setIndex, gameIndex, pointIndex)]
def select_clips(data):
    selected_points_idx = []
    
    def on_set_select(set_idx):
        global selected_set
        selected_set = set_idx
        
        # Clear the games and points lists
        games.delete(0, tk.END)
        points.delete(0, tk.END)
        
        # Populate the games list
        for game_idx in range(len(data[selected_set])):
            games.insert(tk.END, f"game {game_idx+1}")
        
    def on_game_select(game_idx):
        global selected_set, selected_game
        
        selected_game = game_idx
        
        # Clear the points list
        points.delete(0, tk.END)
        
        # Populate the points list
        for point_idx in range(len(data[selected_set][selected_game])):
            pointStartTime = data[selected_set][selected_game][point_idx][0].classicTimestamp()
            pointStopTime = data[selected_set][selected_game][point_idx][1].classicTimestamp()
            points.insert(tk.END, f"{point_idx+1}. [{pointStartTime} - {pointStopTime}]")

        
    def on_point_select(point_idx):
        global selected_set, selected_game
        newPoint = (selected_set, selected_game, point_idx)
        if not newPoint in selected_points_idx:
            selected_points_idx.append((selected_set, selected_game, point_idx))
        
        chosen_points.delete(0, tk.END)
        for (set_idx, game_idx, point_idx) in selected_points_idx:
            pointStartTime = data[set_idx][game_idx][point_idx][0].classicTimestamp()
            pointStopTime = data[set_idx][game_idx][point_idx][1].classicTimestamp()
            chosen_points.insert(tk.END, f"[{pointStartTime} - {pointStopTime}]")
                
            # pointStartTime = data[][selected_game][point_idx][0].classicTimestamp()
            # pointStopTime = data[selected_set][selected_game][point_idx][1].classicTimestamp()
            # chosen_points.insert(tk.END, f"Set{point[0] + 1}/Game{point[1] + 1}/Point{point[2] + 1}")
        
    def on_okay():
        print("Destroying root; Exiting")
        root.destroy()
    
    root = tk.Tk()
    root.title("Select Points")
    
    # Create the sets, games, and points lists
    sets = tk.Listbox(root, width=15, height=25)
    sets_label = tk.Label(root, text="SETS")
    games = tk.Listbox(root, width=15, height=25)
    games_label = tk.Label(root, text="GAMES")
    points = tk.Listbox(root, width=25, height=25)
    points_label = tk.Label(root, text="POINTS")
    chosen_points = tk.Listbox(root, width=25, height=25)
    chosen_points_label = tk.Label(root, text="SELECTED POINTS")

 
    # Populate the sets list and select the first set by default
    for set_idx in range(len(data)):
        sets.insert(tk.END, f"set {set_idx+1}")

    selected_set = 0
    sets.select_set(selected_set)
    
    # Bind the set select event
    sets.bind("<<ListboxSelect>>", lambda e: on_set_select(sets.curselection()[0]))
    
    # Bind the game select event
    games.bind("<<ListboxSelect>>", lambda e: on_game_select(games.curselection()[0]))
    
    # Bind the point select event
    points.bind("<<ListboxSelect>>", lambda e: on_point_select(points.curselection()[0]))
    

    # Pack the widgets
    sets_label.pack(padx=15, pady=15,side=tk.LEFT)
    sets.pack(padx=15, pady=15,side=tk.LEFT)
    games_label.pack(padx=5, pady=5,side=tk.LEFT)
    games.pack(padx=5, pady=5,side=tk.LEFT)
    points_label.pack(padx=5, pady=5,side=tk.LEFT)
    points.pack(padx=5, pady=5,side=tk.LEFT)
    chosen_points_label.pack(padx=5, pady=5,side=tk.LEFT)
    chosen_points.pack(padx=15, pady=15,side=tk.LEFT)


    # Create and pack the "OK" button
    okay_button = tk.Button(root, text="OK", command=on_okay)
    okay_button.pack(pady=5,side=tk.LEFT)

    
    # Start the event loop
    root.mainloop()
    
    return selected_points_idx


# This function generates the highlight reel from the selected clips
def generateReel(src_path: str, dst_path: str, subclipIntervals: list):
    # Load the video
    video = VideoFileClip(src_path)

    # Define the subclip start and end times
    # subclip_times = [(10, 20), (30, 40), (50, 60)]

    # Create a list of subclips
    subclips = []
    for start, end in subclipIntervals:
        subclip = video.subclip(start, end)
        subclips.append(subclip)

    # Concatenate the subclips together
    final_clip = concatenate_videoclips(subclips)

    # Save the final clip
    final_clip.write_videofile(dst_path)

    return


# This function generates the interval of time for the various selected clips
# It sorts out the intervals in increasing order of timestamps
# The first parameter data i.e organized list of sets
# The second parameter is subclipIndices i.e the 
# it returns a sorted list of clip interval tuples i.e [(start_time, stop_time)]
def generateSubclipIntervals(data: list, subclipIndices: list, pointOffset: int = 15, oddGameOffset: int = 70, samplePeriod: int = 5):
    unsortedClipIntervals = []
    for subclipIdx in subclipIndices:
        set_idx = subclipIdx[0]
        game_idx = subclipIdx[1]
        point_idx = subclipIdx[2]
        startFrame = data[set_idx][game_idx][point_idx][0] 
        stopFrame = data[set_idx][game_idx][point_idx][1]
        (startTime, stopTime) = findEffectiveClipInterval(startFrame=startFrame, stopFrame=stopFrame, pointOffset=pointOffset, oddGameOffset=oddGameOffset, samplePeriod=samplePeriod)
        unsortedClipIntervals.append((startTime , stopTime))
    
    sortedClipIntervals = sorted(unsortedClipIntervals, key=lambda interval : interval[0])
    return sortedClipIntervals


def findEffectiveClipInterval(startFrame: Frame, stopFrame: Frame, pointOffset: int = 15 , oddGameOffset: int = 70, samplePeriod: int = 5):
    start_time = startFrame.timestamp 
    stop_time = stopFrame.timestamp
    is_odd_game = False
    _odd_game_offset = 0
    _point_offset = 0

    try:
        player1GamesWonInSet = int(startFrame.data[0][0]) # Number of games won by player 1 in the current set
        player2GamesWonInSet = int(startFrame.data[1][0]) # Number of games won by player 2 in the current set

        # check if total number of games in set is odd
        # or a new set is starting
        is_odd_game = (player1GamesWonInSet + player2GamesWonInSet) % 2  is 1 or (player1GamesWonInSet == 0 and player2GamesWonInSet == 0)
    except:
        is_odd_game = False

    # if is_odd_game:
    #     # if oddGameOffset -  3 * samplePeriod // 2 >= 0:
    #         _odd_game_offset = oddGameOffset + 3 * samplePeriod // 2
    # else:
    #     # if pointOffset - samplePeriod >= 0:
    #         _point_offset = pointOffset + samplePeriod

    if is_odd_game:
        _odd_game_offset = oddGameOffset
    else:
        _point_offset = pointOffset

    clipInterval = stop_time - start_time

    if _odd_game_offset >= clipInterval:
        if clipInterval > 30:
            _odd_game_offset = clipInterval - 30 
        else:
            _odd_game_offset = 0
    
    if _point_offset >= clipInterval:
        _point_offset = 0
    
    return (start_time + _point_offset + _odd_game_offset, stop_time)


# Calculates and returns the timestamp of a frame given the: Frame rate (fps) and Frame number or count(count)
def calcTimestamp(fps: float, count: int):
    return int (count / fps)       


def run_app():
    print("please select a video file")
    video_path = select_video_file()
    sampledFrames = processFrames(path= video_path, sampleTime= DEFAULT_SAMPLE_TIME, samplePeriod=DEFAULT_SAMPLE_PERIOD)
    print(f"edge_x: {[sampleFrame.data[2] for sampleFrame in sampledFrames]}")
    matchData = detectChangesAndSplitFrames(sampledFrames)
    selected_clips_idx = select_clips(matchData)
    clipIntervals = generateSubclipIntervals(data=matchData, subclipIndices=selected_clips_idx, pointOffset = POINT_TIME_OFFSET, oddGameOffset=ODD_GAME_TIME_OFFSET,samplePeriod=DEFAULT_SAMPLE_PERIOD )
    print(f'clip intervals: {clipIntervals}')
    output_dir_path = select_directory()
    output_path = os.path.join(output_dir_path, "hightlight_reel.mp4")
    generateReel(src_path=video_path, dst_path=output_path, subclipIntervals=clipIntervals)

    print(f"Selected Points indices: {selected_clips_idx}")

    print(f"Number of sets: {len(matchData)}")

    print(f"Number of games: {sum([len(set) for set in matchData])}")

    print(f"Number of points: {sum([ sum([len(game) for game in set]) for set in matchData])}")


run_app()

