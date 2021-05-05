import cv2
import glob
import socket
from mss import mss
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from imgaug import augmenters as iaa
import pygetwindow as gw
import threading
import PySimpleGUI as sg
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
import os.path


# Default Paths
homedir = os.path.expanduser("~")
default_steering_path = homedir + r"\AppData\Local\Programs\SelfDrivingCarApplication\Data\SteeringAngles"
default_frames_path = homedir + r"\AppData\Local\Programs\SelfDrivingCarApplication\Data\Images\Frames"
default_resized_path = homedir + r"\AppData\Local\Programs\SelfDrivingCarApplication\Data\Images\Resized"
default_models_path = homedir + r"\AppData\Local\Programs\SelfDrivingCarApplication\Models"
default_paths_steering = homedir + r"\AppData\Local\Programs\SelfDrivingCarApplication\Data\Paths\Angle_Path.txt"
default_paths_frames = homedir + r"\AppData\Local\Programs\SelfDrivingCarApplication\Data\Paths\Frame_Path.txt"

# Default Variables
default_model_name = "Model_Default"
default_epoch_steps = 200
default_num_epochs = 10
default_batch_train = 150
default_batch_val = 100
default_val_steps = 100

# Global Variables
break_collecting_data = False
break_testing = False
bad_inputs = False


# preprocesses a frame and returns it
def frame_preprocessing(frame):
    frame_YUV = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
    frame_blur = cv2.GaussianBlur(frame_YUV, (5, 5), 0)
    frame_normalized = frame_blur / 255
    return frame_normalized


# saves a list of images to a specified folder
def save_images(list_of_images, path_to_folder):
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    for count, img in enumerate(list_of_images):
        count_str = str(count)
        count_edit = count_str.zfill(5)
        cv2.imwrite(str(path_to_folder) + "/frame" + count_edit + ".png", img)
        print("frame" + count_edit)


def save_angles(list_of_angles, path_to_file):
    with open(path_to_file, 'w') as file:
        for angle in list_of_angles:
            file.write("%f\n" % angle)


# receives steering angle and image data from simulator
def data_collection(window, folder_for_original_images, folder_for_steering_angles, file_name_for_steering_angles,
                    folder_for_small_images, list_of_angles_paths, list_of_frames_paths, remove_excess):
    # Server
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 5005))

    # Steering Angles
    steering_angles = []
    # Screen Capture
    frames = []
    sct = mss()
    # collecting data
    while True:
        bounding_box = window_size_and_pos()
        data, address = s.recvfrom(1024)
        data = data.decode("utf-8")
        if data != "paused":
            steering_angles.append(float(data))
            print(data)
            screenshot_img = sct.grab(bounding_box)
            img = Image.frombytes("RGB", screenshot_img.size, screenshot_img.bgra, "raw", "BGRX")
            open_cv_image = np.array(img)
            # Convert RGB to BGR
            open_cv_image = open_cv_image[:, :, ::-1]
            frames.append(open_cv_image)
        if break_collecting_data is True:
            break

    if remove_excess is True:
        print("Removing Excess")
        steering_angles, frames = remove_excess_data(steering_angles, frames, 0, 0.15)

    print("Saving Original Images")
    save_images(frames, folder_for_original_images)
    print("Saving Steering Angles")
    save_angles(steering_angles, folder_for_steering_angles + "/" + file_name_for_steering_angles + ".txt")
    print("Saving Resized Images")
    save_images(crop_and_resize_frames(frames), folder_for_small_images)
    print("Adding paths to text files")
    add_paths_to_list_of_paths(list_of_angles_paths,
                               folder_for_steering_angles + "/" + file_name_for_steering_angles + ".txt", "w")
    add_paths_to_list_of_paths(list_of_frames_paths, folder_for_small_images, "w")
    print("All data is saved, the window can now be closed")


def get_lists(path_to_list_of_paths):
    with open(path_to_list_of_paths) as f:
        list_of_paths = [line.rstrip() for line in f]
    return list_of_paths


def add_paths_to_list_of_paths(list_of_paths, path, add_rewrite):
    with open(list_of_paths, add_rewrite) as file:
        file.write(path + "\n")


# reads file/folder containing steering angles * 2 and images, returns a list of both
def import_data(path_to_steering_angles, path_to_images):
    with open(path_to_steering_angles) as f:
        steering_angles = [line.rstrip() for line in f]
    filenames = glob.glob(str(path_to_images) + "/*.png")
    frames = [cv2.imread(f) for f in filenames]
    steering_angles_data_float = list(map(float, steering_angles))
    return steering_angles_data_float, frames


# plots a histogram and displays it showing the distribution of steering angles the model is trained on
def plot_histogram(num_bins, steering_angles):
    hist, bins = np.histogram(steering_angles, num_bins)
    center = (bins[:-1] + bins[1:]) * 0.5  # center the bins to 0
    plt.bar(center, hist, width=0.05)
    plt.plot((np.min(steering_angles), np.max(steering_angles)), (0, 0))
    plt.title("Distribution of Steering Angles")
    plt.xlabel("Normalised Steering Angle")
    plt.ylabel("Number of Steering Angles")
    plt.xlim([-1, 1])
    plt.show()


def crop_and_resize_frames(list_of_frames):
    frames_finished = []
    for frame in list_of_frames:
        h, w, c = np.shape(frame)
        frame_cropped = frame[int(h / 2) + 50:h - 145, 20:w - 20].copy()
        frame_resize = cv2.resize(frame_cropped, (200, 66), cv2.INTER_AREA)
        frames_finished.append(frame_resize)
    return frames_finished


def load_images_from_folder(path_to_folder):
    filenames = glob.glob(str(path_to_folder) + "/*.png")
    images = [cv2.imread(f) for f in filenames]
    return images


def augment_image(frame, steering):
    img = frame
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.2, 1.2))
        img = brightness.augment_image(img)
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering


def batch_gen(frames, steering_list, batch_size, train_flag):
    while True:
        img_batch = []
        steering_batch = []

        for i in range(batch_size):
            index = random.randint(0, len(frames) - 1)
            if train_flag:
                img, steering = augment_image(frames[index], steering_list[index])
            else:
                img = frames[index]
            steering = steering_list[index]
            img = frame_preprocessing(img)
            img_batch.append(img)
            steering_batch.append(steering)
        yield (np.array(img_batch), np.array(steering_batch))


def window_size_and_pos():
    win = gw.getWindowsWithTitle('CarDrivingSimulator')
    if len(win) > 0:
        win = win[0]
        w, h = win.size
        t = win.top
        le = win.left
        bounding_box = {'top': t, 'left': le, 'width': w, 'height': h}
    else:
        bounding_box = {'top': 200, 'left': 200, 'width': 1024, 'height': 600}

    return bounding_box


def train_model_thread(path_to_trained_model, list_of_paths_to_steering_angles, list_of_paths_of_images, window,
                       epoch_steps, num_epochs, images_per_batch_train, images_per_batch_validation, val_steps):
    threading.Thread(target=train_model, args=(path_to_trained_model, list_of_paths_to_steering_angles,
                                               list_of_paths_of_images, window, epoch_steps, num_epochs,
                                               images_per_batch_train, images_per_batch_validation, val_steps,),
                     daemon=True).start()


def data_collection_thread(window, folder_for_original_images, folder_for_steering_angles,
                           file_name_for_steering_angles, folder_for_small_images, list_of_angles_paths,
                           list_of_frames_paths, remove_excess):
    threading.Thread(target=data_collection, args=(window, folder_for_original_images, folder_for_steering_angles,
                                                   file_name_for_steering_angles, folder_for_small_images,
                                                   list_of_angles_paths, list_of_frames_paths, remove_excess),
                     daemon=True).start()


def test_model_simulator_thread(path_to_model, window):
    threading.Thread(target=test_model_simulator, args=(path_to_model, window,), daemon=True).start()


def remove_excess_data(angles, images, number, maximum_percent):
    angles = [angle * 2 for angle in angles]
    plot_histogram(40, angles)
    count = angles.count(0)
    to_remove = []

    for j, i in enumerate(angles):
        if number - 0.03 < i < number + 0.03 and (
                (count - len(to_remove)) / (len(angles) - len(to_remove))) >= maximum_percent:
            to_remove.append(j)

    to_remove.reverse()
    for i in to_remove:
        del angles[i]
        del images[i]
    #plot_histogram(40, angles)
    angles = [angle / 2 for angle in angles]
    return angles, images


def collecting_data(folder_for_original_images, folder_for_steering_angles, file_name_for_steering_angles,
                    folder_for_small_images, list_of_angles_paths, list_of_frames_paths, remove_excess):
    global break_collecting_data
    # Define the window's contents
    layout = [[sg.Text("Data Collection")],
              [sg.Text("Press Go to start the connection")],
              [sg.Output(size=(150, 50))],
              [sg.Button('Go')],
              [sg.Button('Stop')]]

    # Create the window
    window = sg.Window('Data Collection', layout)  # Part 3 - Window Definition

    while True:  # Event Loop
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == 'Go' and break_collecting_data is not True:
            print('Starting Collection')
            data_collection_thread(window, folder_for_original_images, folder_for_steering_angles,
                                   file_name_for_steering_angles, folder_for_small_images,
                                   list_of_angles_paths, list_of_frames_paths, remove_excess)

        if event == 'Stop':
            break_collecting_data = True
            print("Stopping Data Collection, please wait before closing the window")

    window.close()


def testing_cnn(path_to_model):
    layout = [[sg.Output(size=(150, 50))],
              [sg.Button('Go')], [sg.Button('Stop')]]

    window = sg.Window('Testing CNN', layout)
    global break_testing

    while True:  # Event Loop
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == 'Go':
            print('Starting Testing')
            test_model_simulator_thread(path_to_model, window)
        if event == 'Stop':
            break_testing = True
            break
    window.close()


def training_cnn(path_to_trained_model, path_to_list_of_steering_paths, path_to_list_of_frames_paths, epoch_steps,
                 num_epochs, images_per_batch_train, images_per_batch_validation, val_steps):
    layout = [[sg.Output(size=(150, 50))],
              [sg.Button('Go')]]

    window = sg.Window('Training CNN', layout)

    while True:  # Event Loop
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        if event == 'Go':
            print('Starting Training')
            train_model_thread(path_to_trained_model, get_lists(path_to_list_of_steering_paths),
                               get_lists(path_to_list_of_frames_paths), window, epoch_steps,
                               num_epochs, images_per_batch_train, images_per_batch_validation, val_steps)
    window.close()


def gui():
    # Define variables
    folder_original = None
    folder_small = None
    folder_angles = None
    angles_name = None
    model_name = None
    model_folder = None
    angle_path = None
    frames_path = None
    trained_model_path = None
    epoch_steps = None
    num_epochs = None
    batch_train = None
    batch_val = None
    val_steps = None

    # Themes
    sg.theme("DarkTeal11")
    sg.DEFAULT_FONT = ("Helvetica", 10)
    sg.theme_button_color(color=("White", "Gray"))

    # Define the window's contents
    collecting_data_window = [[sg.Text("Data Collection Settings:")],
                              [sg.HSeparator()],
                              [sg.Text("Folder to store original images:")],
                              [sg.In(size=(25, 1), enable_events=True, key="-Original-"),
                               sg.FolderBrowse()],
                              [sg.Text("Folder to store cropped and resized images:")],
                              [sg.In(size=(25, 1), enable_events=True, key="-Small-"),
                               sg.FolderBrowse()],
                              [sg.Text("Folder to store steering angles:")],
                              [sg.In(size=(25, 1), enable_events=True, key="-Angles-"),
                               sg.FolderBrowse()],
                              [sg.Text("Path to list of frames paths:")],
                              [sg.In(size=(25, 1), enable_events=True, key="-ListFrames1-"),
                               sg.FileBrowse()],
                              [sg.Text("Path to list of angles paths:")],
                              [sg.In(size=(25, 1), enable_events=True, key="-ListAngles1-"),
                               sg.FileBrowse()],
                              [sg.Text("File name for steering angles:")],
                              [sg.In(size=(25, 1), enable_events=True, key="-AnglesName-")],
                              [sg.HSeparator()],
                              [sg.Button("Data Collection")],
                              [sg.Checkbox("Default Settings", key="Def_Data"),
                               sg.Checkbox("Remove Excess Data", key="RemoveExcess")]]

    training_cnn_window = [[sg.Text("Training CNN Settings:")],
                           [sg.HSeparator()],
                           [sg.Text("Folder to store trained model:")],
                           [sg.In(size=(25, 1), enable_events=True, key="-ModelFolder-"),
                            sg.FolderBrowse()],
                           [sg.Text("Path to list of paths to steering angles:")],
                           [sg.In(size=(25, 1), enable_events=True, key="-ListAngles-"),
                            sg.FileBrowse()],
                           [sg.Text("Path to list of paths to small frames:")],
                           [sg.In(size=(25, 1), enable_events=True, key="-ListFrames-"),
                            sg.FileBrowse()],
                           [sg.Text("Name of model:")],
                           [sg.In(size=(25, 1), enable_events=True, key="-ModelName-")],
                           [sg.Text("Input number of steps per epoch: ")],
                           [sg.In(size=(25, 1), enable_events=True, key="-EpochSteps-")],
                           [sg.Text("Input number of epochs:")],
                           [sg.In(size=(25, 1), enable_events=True, key="-NumEpochs-")],
                           [sg.Text("Input number of training images per batch:")],
                           [sg.In(size=(25, 1), enable_events=True, key="-TrainBatch-")],
                           [sg.Text("Input number of validation images per batch:")],
                           [sg.In(size=(25, 1), enable_events=True, key="-ValBatch-")],
                           [sg.Text("Input number of steps per validation:")],
                           [sg.In(size=(25, 1), enable_events=True, key="-ValSteps-")],
                           [sg.HSeparator()],
                           [sg.Button("Training CNN")],
                           [sg.Checkbox("Default Settings", key="Def_Train")]]

    testing_cnn_window = [[sg.Text("Testing CNN Settings:")],
                          [sg.HSeparator()],
                          [sg.Text("Path to trained model:")],
                          [sg.In(size=(25, 1), enable_events=True, key="-TrainedModel-"),
                           sg.FileBrowse()],
                          [sg.HSeparator()],
                          [sg.Button("Testing CNN")]]

    layout = [[sg.Column([[sg.Text("Self Car Driving Simulator Application", justification="center",
                                   font=("Helvetica", 25))]], element_justification="center", justification="center")],
              [sg.HSeparator()],
              [sg.HSeparator()],
              [sg.VSeperator(),
               sg.Column(collecting_data_window, vertical_alignment="top", element_justification='center'),
               sg.VSeperator(),
               sg.Column(training_cnn_window, vertical_alignment="top", element_justification='center'),
               sg.VSeperator(),
               sg.Column(testing_cnn_window, vertical_alignment="top", element_justification='center'),
               sg.VSeperator()],
              [sg.HSeparator()],
              [sg.HSeparator()]]

    # Create the window
    window = sg.Window("Self Driving Car Simulator with Testing and Training", layout,
                       icon=r'C:\Users\morga\PycharmProjects\CNNSelfDrivingCar\Sim_Icon.ico')

    # Display and interact with the Window using an Event Loop
    while True:
        event, values = window.read()
        if event == "-Original-":
            folder_original = values["-Original-"]
        if event == "-Small-":
            folder_small = values["-Small-"]
        if event == "-Angles-":
            folder_angles = values["-Angles-"]
        if event == "-AnglesName-":
            angles_name = values["-AnglesName-"]
        if event == "-ModelName-":
            model_name = values["-ModelName-"]
        if event == "-ModelFolder-":
            model_folder = values["-ModelFolder-"]
        if event == "-ListAngles-":
            angle_path = values["-ListAngles-"]
        if event == "-ListFrames-":
            frames_path = values["-ListFrames-"]
        if event == "-ListAngles1-":
            angle_path = values["-ListAngles1-"]
        if event == "-ListFrames1-":
            frames_path = values["-ListFrames1-"]
        if event == "-TrainedModel-":
            trained_model_path = values["-TrainedModel-"]
        if event == "-EpochSteps-":
            if values['-EpochSteps-'] and values['-EpochSteps-'][-1] not in ('0123456789.'):
                window['-EpochSteps-'].update(values['-EpochSteps-'][:-1])
            else:
                if values['-EpochSteps-'] != "":
                    epoch_steps = int(values['-EpochSteps-'])
        if event == "-NumEpochs-":
            if values['-NumEpochs-'] and values['-NumEpochs-'][-1] not in ('0123456789.'):
                window['-NumEpochs-'].update(values['-NumEpochs-'][:-1])
            else:
                if values['-NumEpochs-'] != "":
                    num_epochs = int(values['-NumEpochs-'])
        if event == "-TrainBatch-":
            if values['-TrainBatch-'] and values['-TrainBatch-'][-1] not in ('0123456789.'):
                window['-TrainBatch-'].update(values['-TrainBatch-'][:-1])
            else:
                if values['-TrainBatch-'] != "":
                    batch_train = int(values['-TrainBatch-'])
        if event == "-ValBatch-":
            if values['-ValBatch-'] and values['-ValBatch-'][-1] not in ('0123456789.'):
                window['-ValBatch-'].update(values['-ValBatch-'][:-1])
            else:
                if values['-ValBatch-'] != "":
                    batch_val = int(values['-ValBatch-'])
        if event == "-ValSteps-":
            if values['-ValSteps-'] and values['-ValSteps-'][-1] not in ('0123456789.'):
                window['-ValSteps-'].update(values['-ValSteps-'][:-1])
            else:
                if values['-ValSteps-'] != "":
                    val_steps = int(values['-ValSteps-'])
        if event == "Data Collection":
            process = 0
            remove_excess = values["RemoveExcess"]
            default_collect = values["Def_Data"]
            window.close()
            if default_collect is True:
                return [process, default_frames_path, default_steering_path, "SteeringAngles.txt", default_resized_path,
                        default_paths_steering, default_paths_frames, remove_excess]
            else:
                return [process, folder_original, folder_angles, angles_name, folder_small, angle_path, frames_path,
                        remove_excess]
        if event == "Training CNN":
            process = 1
            window.close()
            default_train = values["Def_Train"]
            if default_train is True:
                return [process, default_models_path, default_model_name, default_paths_steering, default_paths_frames,
                        default_epoch_steps, default_num_epochs, default_batch_train, default_batch_val,
                        default_val_steps]
            else:
                return [process, model_folder, model_name, angle_path, frames_path, epoch_steps, num_epochs,
                        batch_train, batch_val, val_steps]
        if event == "Testing CNN":
            process = 2
            window.close()
            return [process, trained_model_path]
        # See if user wants to quit or window was closed
        if event == sg.WINDOW_CLOSED:
            process = 3
            window.close()
            return [process]


# gets image data from the simulator and sends back predicted steering angle from model used
def test_model_simulator(path_to_model, window):
    # Import trained model
    trained_model = keras.models.load_model(path_to_model)
    # Server
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Screen Capture
    sct = mss()
    while True:
        bounding_box = window_size_and_pos()
        screenshot_img = sct.grab(bounding_box)
        input_image = Image.frombytes("RGB", screenshot_img.size, screenshot_img.bgra, "raw", "BGRX")
        open_cv_image = np.array(input_image)
        open_cv_image = open_cv_image[:, :, ::-1]
        h, w, c = np.shape(open_cv_image)
        frame_cropped = open_cv_image[int(h / 2) + 50:h - 145, 20:w - 20].copy()
        frame_resize = cv2.resize(frame_cropped, (200, 66), cv2.INTER_AREA)
        frame_processed = frame_preprocessing(frame_resize)
        test_image_array = np.array([frame_processed])
        prediction = round((float(trained_model.predict(test_image_array)) / 2), 3)
        print(str(prediction))
        window.write_event_value('-Angle-', prediction)
        prediction = str(prediction).encode('utf8')
        s.sendto(prediction, ("127.0.0.1", 1234))
        if break_testing is True:
            window.write_event_value('-Angle-', "Stopping Testing")
            break


def train_model(path_to_trained_model, list_of_paths_to_steering_angles, list_of_paths_of_images, window,
                epoch_steps, num_epochs, images_per_batch_train, images_per_batch_validation, val_steps):
    print("Using the following settings.." + "\n |number of epochs: " + str(num_epochs) + "\n |epoch steps: " +
          str(epoch_steps) + "\n |train images per batch: " + str(images_per_batch_train) +
          "\n |val images per batch: " + str(images_per_batch_validation) + "\n |val steps: " + str(val_steps))
    # get model
    print("Getting Model...")
    model = model_maker_nvidia()
    # lists of model inputs
    frames = []
    steering_angles_float = []
    for steering_angle_path, image_path in zip(list_of_paths_to_steering_angles, list_of_paths_of_images):
        steering_angles_float_temp, frames_temp = import_data(steering_angle_path, image_path)
        frames += frames_temp
        steering_angles_float += steering_angles_float_temp

    # plots histogram of steering angles
    print("Plotting Histogram")
    steering_angles_float = [angle * 2 for angle in steering_angles_float]
    plot_histogram(31, steering_angles_float)

    # convert model inputs to array
    steering_angles_float_array = np.array(steering_angles_float)
    frames = np.array(frames)

    print("Performing Test Train Split")
    # perform test train split
    x_train, x_valid, y_train, y_valid = train_test_split(frames, steering_angles_float_array, test_size=0.2,
                                                          shuffle=1)
    print("Total Training Images: ", len(x_train))
    print("Total Validation Images: ", len(x_valid))
    print("Training Model")
    call = callbacks.ProgbarLogger()
    window.write_event_value('-THREAD PROGRESS-', str(call))
    history = model.fit(batch_gen(x_train, y_train, images_per_batch_train, 1), steps_per_epoch=epoch_steps,
                        epochs=num_epochs, validation_data=batch_gen(x_valid, y_valid, images_per_batch_validation, 0),
                        validation_steps=val_steps, callbacks=[call], verbose=2)
    # saves model
    print("Saving Model")
    model.save(path_to_trained_model)
    # plots training results
    print("Showing Results")
    plt.plot(history.history['loss'], marker='x')
    plt.plot(history.history['val_loss'], marker='x')
    plt.legend(['Training', 'Validation'])
    plt.xticks(range(0, num_epochs))
    plt.title('Training and Validation Loss Graph')
    plt.xlabel('Epochs')
    plt.ylabel("Mean Squared Error")
    plt.show()

    print("The window can now be closed")


# creates CNN model architecture and returns compiled model.
def model_maker_nvidia():
    # Model
    model = Sequential()
    # Add Layers Convolutional
    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), (1, 1), activation='elu'))
    model.add(Convolution2D(64, (3, 3), (1, 1), activation='elu'))
    # Add Layer Flatten
    model.add(Flatten())
    # Add Layer Dense
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    # Compile Model
    model.compile(Adam(lr=0.0001), loss='mse')

    return model
