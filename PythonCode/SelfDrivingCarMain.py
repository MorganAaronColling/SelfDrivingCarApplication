import subprocess
from SelfDrivingCarFunctions import collecting_data, gui, testing_cnn, training_cnn
import SelfDrivingCarFunctions
import PySimpleGUI as sg
import os.path

# Default simulator path
homedir = os.path.expanduser("~")
default_simulator_path = homedir + r"\AppData\Local\Programs\SelfDrivingCarApplication\CarDrivingSimulator\CarDrivingSimulator.exe"

returned_list = ["NotNone"]
complete = False
while not complete:
    SelfDrivingCarFunctions.bad_inputs = False
    if returned_list[0] != "NotNone":
        for i in returned_list:
            if i is None:
                SelfDrivingCarFunctions.bad_inputs = True
                sg.popup('Fill in all relevant fields')
                break
        if SelfDrivingCarFunctions.bad_inputs is False:
            try:
                if returned_list[0] == 0:
                    subprocess.Popen(default_simulator_path)
                    collecting_data(returned_list[1], returned_list[2], returned_list[3], returned_list[4],
                                    returned_list[5], returned_list[6], returned_list[7])
                    returned_list = ["NotNone"]
                elif returned_list[0] == 1:
                    training_cnn(returned_list[1] + "/" + returned_list[2] + ".h5", returned_list[3], returned_list[4],
                                 returned_list[5], returned_list[6], returned_list[7], returned_list[8], returned_list[9])
                    returned_list = ["NotNone"]
                elif returned_list[0] == 2:
                    subprocess.Popen(default_simulator_path)
                    testing_cnn(returned_list[1])
                    returned_list = ["NotNone"]
                elif returned_list[0] == 3:
                    complete = True
            except:
                sg.popup('Invalid Arguments')
    if complete:
        break
    returned_list = gui()
