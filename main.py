import tkinter as tk
from tkinter import filedialog
import time
import os
import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import keras_ocr
import sys

sys.path.insert(1, 'src')

import frame as fr
import moviepy.editor as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
import ocr

# Load the Keras-OCR model

# pipeline = keras_ocr.pipeline.Pipeline()

SET_CHANGE_THRESH = 10
POINT_TIME_OFFSET = 15
ODD_GAME_TIME_OFFSET = 90
DEFAULT_SAMPLE_TIME = 120

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

    frameCount = 0
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


def kerasocrPredict(images, isSingleChannel=True):

    if isSingleChannel == True:
        images = [cv2.merge((image, image, image)) for image in images]

    predictions = pipeline.recognize(images)

    return ["".join([prediction[0] for prediction in predictions[i]]) for i in range(len(predictions)) ]


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

                cv2.imshow(f"Sample: {sampleCount} of {numberOfSamples}",frame)
                cv2.waitKey(10)
                newFrame = fr.Frame(
                    count=frameCount,
                    timestamp= calcTimestamp(fps, frameCount),
                    fps=fps,
                    data=ocr.extractFrameData(frame),
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

def hasChanged_set(prevFrame: fr.Frame, currentFrame: fr.Frame):
    prevEdgeXCoord = prevFrame.data[2]
    currentEdgeXCoord = currentFrame.data[2]
    if abs(prevEdgeXCoord - currentEdgeXCoord) >= SET_CHANGE_THRESH:
        return True
    else:
        return False

def hasChanged_game(prevFrame: fr.Frame, currentFrame: fr.Frame):
    prevTopleftDigits = prevFrame.data[0][0]
    prevBottomleftDigits = prevFrame.data[1][0]
    currentTopleftDigits = currentFrame.data[0][0]
    currentBottomleftDigits = currentFrame.data[1][0]
    if prevTopleftDigits != currentTopleftDigits or prevBottomleftDigits != currentBottomleftDigits:
        return True
    else:
        return False

def hasChanged_point(prevFrame: fr.Frame, currentFrame: fr.Frame):
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

            # ADD THE CLOSING FREAME OF THE LASTEST POINT
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

            # ADD THE CLOSING FREAME OF THE LAST POINT
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
def generateSubclipIntervals(data: list, subclipIndices: list, pointOffset: int = 0):
    unsortedClipIntervals = []
    for subclipIdx in subclipIndices:
        set_idx = subclipIdx[0]
        game_idx = subclipIdx[1]
        point_idx = subclipIdx[2]
        start_time = data[set_idx][game_idx][point_idx][0].timestamp 
        stop_time = data[set_idx][game_idx][point_idx][1].timestamp

        if start_time + pointOffset - stop_time > 5:
            unsortedClipIntervals.append((start_time + pointOffset, stop_time))
        else:
            unsortedClipIntervals.append((start_time , stop_time))
    
    sortedClipIntervals = sorted(unsortedClipIntervals, key=lambda interval : interval[0])
    return sortedClipIntervals


# Calculates and returns the timestamp of a frame given the: Frame rate (fps) and Frame number or count(count)
def calcTimestamp(fps: float, count: int):
    return int (count / fps)       


def run_app():
    print("please select a video file")
    video_path = select_video_file()
    sampledFrames = processFrames(path= video_path, sampleTime= 600)
    matchData = detectChangesAndSplitFrames(sampledFrames)
    selected_clips_idx = select_clips(matchData)
    clipIntervals = generateSubclipIntervals(data=matchData, subclipIndices=selected_clips_idx, pointOffset = POINT_TIME_OFFSET )
    print(f'clip intervals: {clipIntervals}')
    output_dir_path = select_directory()
    output_path = os.path.join(output_dir_path, "hightlight_reel.mp4")
    generateReel(src_path=video_path, dst_path=output_path, subclipIntervals=clipIntervals)

    print(f"Selected Points indices: {selected_clips_idx}")

    print(f"Number of sets: {len(matchData)}")

    print(f"Number of games: {sum([len(set) for set in matchData])}")

    print(f"Number of points: {sum([ sum([len(game) for game in set]) for set in matchData])}")


run_app()