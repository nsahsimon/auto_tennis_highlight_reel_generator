from copy import deepcopy
import tkinter as tk
from tkinter import ttk
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
import math
import shutil
from multiprocessing import Pool

SET_CHANGE_THRESH = 10
POINT_TIME_OFFSET = 15
ODD_GAME_TIME_OFFSET = 70
DEFAULT_SAMPLE_TIME = 300
DEFAULT_SAMPLE_PERIOD = 10
LOG_FILENAME = "logs.txt"
video_path = None


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

    def setTimestamp(self, new_timestamp):
        self.timestamp = new_timestamp


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


def sortTexts(txts, boxes):
    txt_boxes = []
    for i, box in enumerate(boxes):
        txt_boxes.append((txts[i],box))

    # sort vertically
    txt_boxes.sort(key=lambda txt_box: txt_box[1][0][1])

    # sort the first half horizontally
    txt_boxes[:2]  = sorted(txt_boxes[:2] , key=lambda txt_box: txt_box[1][0][0]) # Sorting in ascending order

    # sort the second half horizontally
    txt_boxes[2:] = sorted(txt_boxes[2:], key=lambda txt_box: txt_box[1][0][0]) # Sorting in ascending order

    return [txt_box[0] for txt_box in txt_boxes]


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
    txts = sortTexts(results['txts'], results['boxes'])
    print(txts)
    up = None; down = None
    if len(txts) == 4:
        up = [txts[0] , txts[1]]
        down = [txts[2], txts[3]]
    return [up, down, edge_x]


# test function
def test():
    file_names = os.listdir(path='images/')
    for i, name in enumerate(file_names):
        full_path = os.path.join('images/', name)
        image = cv2.imread(full_path)
        data = extractFrameData(image, ocr=OCR)
        print(f"{i + 1}: {name}")
        print(f" up: {data[0]} \n down: {data[1]} \n edge_x: {data[2]} \n")
        cv2.imshow("image", image)
        cv2.waitKey()
        cv2.destroyAllWindows()



# UI functions

def get_user_input():

    user_input = {}
    # global progress_bar; progress_bar = None
    
    # Create the main window
    global window ; window = tk.Tk()
    window.title("Input Information")
    # Set the size of the window to 500x300 pixels
    window.geometry("512x640")

    # Get source video file

    src_video_entry = tk.Entry(window, textvariable=tk.StringVar(value="No file selected"), width=50, state=tk.DISABLED)
    src_video_entry.pack()
    def choose_src_file():
        video_file_path = select_video_file()
        user_input['src_file'] = video_file_path
        src_video_entry.config(textvariable=tk.StringVar(value=f"{video_file_path}"))

    choose_src_button = tk.Button(window, text="Choose a video", command=choose_src_file)
    choose_src_button.pack(pady=10)


    # Create the labels and text fields for each input
    point_offset_label = tk.Label(window, text="By how many seconds should the start of each point be offset? (Seconds): ")
    point_offset_label.pack()
    point_offset_entry = tk.Entry(window, textvariable=tk.StringVar(value=f"{POINT_TIME_OFFSET}"))
    point_offset_entry.pack(pady=20)

    odd_offset_label = tk.Label(window, text="By how many seconds should the start of each odd game offset? (Seconds): ")
    odd_offset_label.pack()
    odd_offset_entry = tk.Entry(window, textvariable=tk.StringVar(value=f"{ODD_GAME_TIME_OFFSET}"))
    odd_offset_entry.pack(pady=20)

    video_duration_label = tk.Label(window, text="What Length of the video do you want to be processed? (Seconds): ")
    video_duration_label.pack()
    video_duration_entry = tk.Entry(window, textvariable=tk.StringVar(value=f"{DEFAULT_SAMPLE_TIME}"))
    video_duration_entry.pack(pady=20)

    sampling_period_label = tk.Label(window, text="After how many seconds should the video frames be sampled? (Seconds) : ")
    sampling_period_label.pack()
    sampling_period_label = tk.Label(window, text="(Recommended: 7 secs): ")
    sampling_period_label.pack()
    sampling_period_entry = tk.Entry(window, textvariable=tk.StringVar(value=f"{DEFAULT_SAMPLE_PERIOD}"))
    sampling_period_entry.pack(pady=20)

    dst_video_entry = tk.Entry(window, textvariable=tk.StringVar(value="No Folder selected"), width=50,state=tk.DISABLED )
    dst_video_entry.pack()
    def choose_output_dir():
        output_directory = select_directory()
        user_input['dst_directory'] = output_directory
        dst_video_entry.config(textvariable=tk.StringVar(value=f"{output_directory}"))

    choose_output_button = tk.Button(window, text="Choose output directory", command=choose_output_dir)
    choose_output_button.pack(pady=10)

    def run_app_local():
        # progress_bar.stop()
        # window.destroy()
        run_app(user_input=user_input, progress_bar=progress_bar, window=window)


    def start_background_process():
        # Disable the submit button and text fields
        submit_button.config(state=tk.DISABLED)
        point_offset_entry.config(state=tk.DISABLED)
        odd_offset_entry.config(state=tk.DISABLED)
        video_duration_entry.config(state=tk.DISABLED)
        sampling_period_entry.config(state=tk.DISABLED)

        tk.Label(window, text="Processing video. Please wait....").pack(pady=20)

        # Set up the progress bar
        global progress_bar; progress_bar = ttk.Progressbar(window, orient="horizontal",maximum=100, mode='indeterminate')
        progress_bar.pack(pady=20)
        progress_bar.start()
        window.after(5000, lambda : run_app_local())

        # Start the background process
        # For demonstration purposes, we'll just sleep for 5 seconds here
        # In a real program, you would do some actual work here



    def submit():
        # Get the values from the text fields and convert them to integers
        user_input["point_offset"] = int(point_offset_entry.get())
        if user_input["point_offset"] < 1:
            print("Invalid point offset")
            return
        
        user_input["odd_game_offset"] = int(odd_offset_entry.get())
        if user_input["odd_game_offset"] < 1:
            print("Invalid odd game offset")
            return
        
        user_input["video_duration"] = int(video_duration_entry.get())
        if user_input["video_duration"] < 1:
            print("Invalid video duration")
            return

        user_input["sampling_period"] = int(sampling_period_entry.get())
        if user_input["sampling_period"] < 1:
            print("Invalid sample period")
            return

        if dst_video_entry.get().lower() == "no folder selected" or src_video_entry.get().lower() == "no file selected":
            print("Invalid point video source file or destination folder")
            return

        # Start the background process
        start_background_process()

    # Create the submit button
    submit_button = tk.Button(window, text="Submit", command=submit)
    submit_button.pack()

    # Run the window
    window.mainloop()

    # Return the user input as a dictionary
    return user_input


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
            first_point = data[selected_set][game_idx][0]
            try:
                player1_games = first_point[0].data[0][0]
                player2_games = first_point[0].data[1][0]
                games.insert(tk.END, f"{game_idx+1}. ({player1_games} - {player2_games})")
            except:
                games.insert(tk.END, f"{game_idx+1}. ('Undefined' - 'Undefined')")
            
        
    def on_game_select(game_idx):
        global selected_set, selected_game
        selected_game = game_idx

        # def get_longest_point_idx():
        longest_point_idx = 0
        for point_idx in range(len(data[selected_set][selected_game])):
            point_of_interest = data[selected_set][selected_game][point_idx]
            longest_point = data[selected_set][selected_game][longest_point_idx]

            try:
                duration_of_point_of_interest = point_of_interest[1].timestamp - point_of_interest[0].timestamp
                duration_of_longest_point = longest_point[1].timestamp - longest_point[0].timestamp
                if duration_of_longest_point < duration_of_point_of_interest:
                    longest_point_idx = point_idx
            except:
                print("ERROR: Encountered null point at Set: {selected_set + 1}, Game: {selected_game + 1}, Point: {point_idx + 1} => {point_of_interest}")
                log_data(f" \n\n ERROR: Encountered null point at Set: {selected_set + 1}, Game: {selected_game + 1}, Point: {point_idx + 1} => {point_of_interest}")
                continue

            # return longest_point_idx
        
        # Clear the points list
        points.delete(0, tk.END)
        
        # Populate the points list
        for point_idx in range(len(data[selected_set][selected_game])):
            point_of_interest = data[selected_set][selected_game][point_idx]
            pointStartTime = point_of_interest[0].classicTimestamp()
            pointStopTime = point_of_interest[1].classicTimestamp()
            player1_points = point_of_interest[0].data[0][1]
            player2_points = point_of_interest[0].data[1][1]
            points.insert(tk.END, f"{point_idx+1}. {'**' if point_idx == longest_point_idx else ''} ({player1_points} - {player2_points}) | ({pointStartTime} - {pointStopTime})")

        
    def on_point_select(point_idx):
        global selected_set, selected_game
        newPoint = (selected_set, selected_game, point_idx)
        if not newPoint in selected_points_idx:
            selected_points_idx.append((selected_set, selected_game, point_idx))
        
        chosen_points.delete(0, tk.END)
        for (set_idx, game_idx, point_idx) in selected_points_idx:
            point_of_interest = data[set_idx][game_idx][point_idx]
            pointStartTime = point_of_interest[0].classicTimestamp()
            pointStopTime = point_of_interest[1].classicTimestamp()
            player1_points = point_of_interest[0].data[0][1]
            player2_points = point_of_interest[0].data[1][1]
            chosen_points.insert(tk.END, f"({player1_points} - {player2_points}) | ({pointStartTime} - {pointStopTime})")
                
    
    def on_chosen_point_double_click(event):
        chosen_point_idx = chosen_points.curselection()[0]
        selected_points_idx.pop(chosen_point_idx)

        chosen_points.delete(0, tk.END)
        for (set_idx, game_idx, point_idx) in selected_points_idx:
            point_of_interest = data[set_idx][game_idx][point_idx]
            pointStartTime = point_of_interest[0].classicTimestamp()
            pointStopTime = point_of_interest[1].classicTimestamp()
            player1_points = point_of_interest[0].data[0][1]
            player2_points = point_of_interest[0].data[1][1]
            chosen_points.insert(tk.END, f"({player1_points} - {player2_points}) | ({pointStartTime} - {pointStopTime})")

        
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
    points = tk.Listbox(root, width=30, height=25)
    points_label = tk.Label(root, text="POINTS")
    chosen_points = tk.Listbox(root, width=30, height=25)
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

    # Bind the chosen points delete event
    chosen_points.bind("<Double-1>", on_chosen_point_double_click)
    

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






# Load paddle ocr
OCR = paddleocr.PaddleOCR(lang='en')
def log_data(data):
    with open(LOG_FILENAME, 'a') as file:
        file.write(f"{data} \n")

def del_prev_log():
    if os.path.exists(LOG_FILENAME):
        os.remove(LOG_FILENAME)

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

def processFrames(cam , sampleTime:int=DEFAULT_SAMPLE_TIME, samplePeriod:int=15):
        sampledFrames = []

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
        
            if frameCount == 0:
                log_data(f"Resolution: ({frame.shape[0]} x {frame.shape[1]}) \n")
            
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


                newFrame = Frame(
                    count=frameCount,

                    # calculating timestamp
                    # samplePeriod / 2 is the correction factor
                    timestamp= int(calcTimestamp(fps, frameCount) - samplePeriod / 2) if frameCount > 0 else calcTimestamp(fps, frameCount),
                    fps=fps,
                    data=newFrameData,
                    image = frame
                )

                log_data(f"""{newFrame.data} at ({newFrame.classicTimestamp()}) """)

                print(f"edge_x: {newFrameData[2]}")

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
# It returns an organized list of match data
def detectChangesAndSplitFrames(sampledFrames, fps, sampleTime, pointOffset, oddGameOffset, samplePeriod):
    generic_last_frame = Frame(
                    count=int(sampleTime * fps),
                    timestamp= sampleTime,
                    fps=fps,
                    data=sampledFrames[len(sampledFrames) - 1].data,
                    image= sampledFrames[len(sampledFrames) - 1].image
                )

    sets = [[[[sampledFrames[0], generic_last_frame], ] ] ] # [set[game[point[start_frame, stop_frame]]]
    
    # Organizing the sampledFrames into SETS -> GAMES -> POINTS
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
            latestSet = [[[sampledFrames[i], generic_last_frame]]]
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
            latestGame = [[sampledFrames[i], generic_last_frame]]
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
            latestPoint = [sampledFrames[i], generic_last_frame]
            latestGame.append(latestPoint)

            # Modify the LATEST GAME
            latestSet[len(latestSet) - 1] = latestGame

            # MODIFY THE LAST SET
            sets[len(sets) - 1] = latestSet

            continue
    

    # return sets
    # Add offsets
    sets_with_offset = sets
    is_new_set = False
    is_odd_game = False
    for _set_idx, _set in enumerate(sets):
        # sets_with_offset.append([])
        is_new_set = True
        for _game_idx, _game in enumerate(_set):
            # sets_with_offset[len(sets_with_offset) - 1].append([])
            try:
                player1GamesWonInSet = int(_game[0][0].data[0][0]) # Number of games won by player 1 in the current set
                player2GamesWonInSet = int(_game[0][1].data[1][0]) # Number of games won by player 2 in the current set

                # check if total number of games in set is odd
                is_odd_game = (player1GamesWonInSet + player2GamesWonInSet) % 2  is 1 or (player1GamesWonInSet == 0 and player2GamesWonInSet == 0)
            except:
                is_odd_game = False

            for _point_idx, _point in enumerate(_game):
                startFrame = deepcopy(_point[0])
                stopFrame = deepcopy(_point[1])
                (startTime, stopTime) = findEffectiveClipInterval(startFrame=startFrame, stopFrame=stopFrame, is_new_set=is_new_set, is_odd_game=is_odd_game, pointOffset=pointOffset, oddGameOffset=oddGameOffset, samplePeriod=samplePeriod)
                startFrame.setTimestamp(startTime)
                # stopFrame.setTimestamp(stopTime)
                sets_with_offset[_set_idx][_game_idx][_point_idx] = (startFrame, stopFrame)

                # reset flags
                is_new_set = False
                is_odd_game = False

    return sets_with_offset


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
    final_clip.write_videofile(dst_path, codec='libx264', audio_codec='aac')

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
        startTime = startFrame.timestamp
        stopTime = stopFrame.timestamp
        # (startTime, stopTime) = findEffectiveClipInterval(startFrame=startFrame, stopFrame=stopFrame, pointOffset=pointOffset, oddGameOffset=oddGameOffset, samplePeriod=samplePeriod)
        unsortedClipIntervals.append((startTime , stopTime))
    
    sortedClipIntervals = sorted(unsortedClipIntervals, key=lambda interval : interval[0])
    return sortedClipIntervals


def findEffectiveClipInterval(startFrame: Frame, stopFrame: Frame, is_new_set: bool, is_odd_game: bool, pointOffset: int = 15 , oddGameOffset: int = 70, samplePeriod: int = 5):
    start_time = startFrame.timestamp 
    stop_time = stopFrame.timestamp
    clip_interval = stop_time  - start_time

    _odd_game_offset = 0
    _point_offset = 0
    _new_set_offset = 0


    def output():
        return (start_time + _point_offset + _odd_game_offset + _new_set_offset, stop_time)
 
   
    if is_new_set:
        if clip_interval - oddGameOffset > 15:
            _new_set_offset = oddGameOffset
            return output()
        elif clip_interval > 20:
            _new_set_offset = clip_interval - 20
            return output()
        elif clip_interval - pointOffset > 15:
            _new_set_offset = pointOffset
            return output()
        else:
            return output()
        
    elif is_odd_game:
        if clip_interval - oddGameOffset > 15:
            _odd_game_offset = oddGameOffset
            return output()
        elif clip_interval > 20:
            _odd_game_offset = clip_interval - 20
            return output()
        elif clip_interval - pointOffset > 15:
            _odd_game_offset = pointOffset
            return output()
        else: 
            return output()
        
    else:
        if clip_interval - pointOffset > 10:
            _point_offset = pointOffset
            return output()
        elif clip_interval > 15:
            _point_offset = clip_interval - 20
            return output()
        else:
            return output()
        

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
    
    return (start_time + _point_offset + _odd_game_offset + _new_set_offset, stop_time)


# Calculates and returns the timestamp of a frame given the: Frame rate (fps) and Frame number or count(count)
def calcTimestamp(fps: float, count: int):
    return int (count / fps)       


def run_app(user_input, progress_bar, window):
    del_prev_log()
    sampleTime = user_input['video_duration']
    log_data(f"Sample time: {sampleTime}")
    samplePeriod = user_input['sampling_period']
    log_data(f"Sample period: {samplePeriod}")
    video_path = user_input['src_file']
    log_data(f"Video path: {video_path}")
    output_dir_path = user_input['dst_directory']
    log_data(f"output directory path: {output_dir_path}")
    pointOffset = user_input['point_offset']
    log_data(f"point offset: {pointOffset}")
    oddGameOffset = user_input['odd_game_offset']
    log_data(f"odd game offset: {oddGameOffset}")
    log_data("\n")
    cam = cv2.VideoCapture(video_path)
    if not cam.isOpened():
        print("Unable to open video")
        exit()
    fps = cam.get(cv2.CAP_PROP_FPS)


    # print("please select a video file")
    # video_path = select_video_file()
    sampledFrames = processFrames(cam= cam, sampleTime= sampleTime, samplePeriod=samplePeriod)
    print(f"edge_x: {[sampleFrame.data[2] for sampleFrame in sampledFrames]}")
    matchData = detectChangesAndSplitFrames(sampledFrames=sampledFrames, fps=fps, sampleTime=sampleTime, pointOffset=pointOffset, oddGameOffset=oddGameOffset, samplePeriod=samplePeriod)
    progress_bar.stop()
    window.destroy()
    selected_clips_idx = select_clips(matchData)
    clipIntervals = generateSubclipIntervals(data=matchData, subclipIndices=selected_clips_idx, pointOffset = pointOffset, oddGameOffset=oddGameOffset,samplePeriod=samplePeriod )
    print(f'clip intervals: {clipIntervals}')
    # output_dir_path = select_directory()
    output_path = os.path.join(output_dir_path, "highlight_reel.mp4")
    generateReel(src_path=video_path, dst_path=output_path, subclipIntervals=clipIntervals)

    print(f"Selected Points indices: {selected_clips_idx}")

    print(f"Number of sets: {len(matchData)}")

    print(f"Number of games: {sum([len(set) for set in matchData])}")

    print(f"Number of points: {sum([ sum([len(game) for game in set]) for set in matchData])}")


get_user_input()

