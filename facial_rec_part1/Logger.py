import os 
from termcolor import colored
from datetime import datetime

LogDirectory = "./logs/"
LogFileName = "log.txt"
AppendDateTime = True

class Logger():

    def __init__(self, logDirectory, logFileName, appendDateTime=True):

        self.LogDirectory = logDirectory
        self.AppendDateTime = appendDateTime

        if (not os.path.isdir(LogDirectory)):
            os.mkdir(LogDirectory)
        
        if (os.path.exists(logDirectory + logFileName)):
            self.LogFileName = logFileName

        elif (appendDateTime):

            f = open(LogDirectory + logFileName, 'a+')
            f.write("Log file creation" + '\n')
            f.close()

            self.LogFileName = logFileName + '-' + \
                datetime.now().strftime("%m-%d-%Y-%H:%M:%S")


    def WriteToLog(self, message):

        if (self.AppendDateTime):
            message = datetime.now().strftime("%H:%M:%S") + '\t' + message

        file_name = self.LogDirectory + self.LogFileName
        f = open(file_name, 'a+')
        f.write(message + '\n')
        f.close()

    def Info(self, message):

        self.WriteToLog(message)
        print(colored(message, 'green'))

    def Warn(self, message):

        self.WriteToLog(message)
        print(colored(message, 'yellow'))

    def Error(self, message):

        self.WriteToLog(message)
        print(colored(message, 'red'))

