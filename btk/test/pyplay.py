#!/usr/bin/env python

import os
import Tkinter
import FileDialog
from Tkconstants import *


class mainApp:
    def __init__(self):
        self.tk = Tkinter.Tk()
        topframe = Tkinter.Frame(self.tk, border=4)
        topframe.pack(fill=BOTH, expand=1)
        frame = Tkinter.Frame(topframe, relief=RAISED, border=2)
        frame.pack(side=TOP, fill=X, expand=1)
        label = Tkinter.Label(frame, text="Filename:")
        label.pack(side=LEFT)
        self.filename = Tkinter.StringVar()
        self.filename.set("")
        entry=Tkinter.Entry(frame, textvariable=self.filename)
        entry.pack(side=LEFT, expand=1, fill=X)
        button = Tkinter.Button(frame, text="...", command=self.openFile)
        button.pack(side=LEFT)

        frame = Tkinter.Frame(topframe, relief=RAISED, border=2)
        frame.pack(side=TOP, fill=X, expand=1)
        label = Tkinter.Label(frame, text="Samplingrate (Hz):")
        label.grid(row=0, column=0)
        self.sampling = Tkinter.StringVar()
        self.sampling.set("22050")
        option = Tkinter.OptionMenu(frame, self.sampling,
                                    "48000", "44100",
                                    "22050", "16000",
                                    "11025", "8000")
        option.grid(row=0, column=1)
        
        label = Tkinter.Label(frame, text="data size:")
        label.grid(row=1, column=0)
        self.datasize = Tkinter.StringVar()
        self.datasize.set("16-bit")
        option = Tkinter.OptionMenu(frame, self.datasize, "32-Bit", "16-bit", "8-bit")
        option.grid(row=1, column=1)

        label = Tkinter.Label(frame, text="Encoding:")
        label.grid(row=2, column=0)
        self.encoding = Tkinter.StringVar()
        self.encoding.set("signed linear")
        option = Tkinter.OptionMenu(frame, self.encoding,
                                    "signed linear",
                                    "unsigned linear",
                                    "u-law",
                                    "A-law",
                                    "ADPCM",
                                    "IMA_ADPCM",
                                    "GMS",
                                    "Floating-point")
        option.grid(row=2, column=1)

        label = Tkinter.Label(frame, text="Channels:")
        label.grid(row=3, column=0)
        self.channels = Tkinter.StringVar()
        self.channels.set("1")
        option = Tkinter.OptionMenu(frame, self.channels, "1", "2", "4")
        option.grid(row=3, column=1)

        self.swap = Tkinter.BooleanVar()
        self.swap.set(0)
        check = Tkinter.Checkbutton(frame, text="Swap Bytes", variable=self.swap)
        check.grid(row=4, column=1)

        frame = Tkinter.Frame(topframe, relief=RAISED, border=2)
        frame.pack(side=BOTTOM, fill=X, expand=1)
        button = Tkinter.Button(frame, text="Play", command=self.play)
        button.pack(side=RIGHT)

    def mainloop(self):
        self.tk.mainloop()

    def openFile(self):
        print "Juhu Kinners"
        d = FileDialog.LoadFileDialog(self.tk)
        file = d.go(".", "*")
        if file:
            self.filename.set(file)

    def play(self):
        cmd = "play -t raw "
        cmd += "-r %s " % self.sampling.get()
        size = ""
        if self.datasize.get() == "32-bit":
            size = "-s l "
        elif self.datasize.get() == "16-bit":
            size = "-s w "
        elif self.datasize.get() == "8-bit":
            size = "-s b "
        else: pass
        cmd += size
        enc = ""
        if self.encoding.get() == "signed linear":
            enc = "-f s "
        elif self.encoding.get() == "unsigned linear":
            enc = "-f u "
        elif self.encoding.get() == "u-law":
            enc = "-f U "
        elif self.encoding.get() == "A-law":
            enc = "-f A "
        elif self.encoding.get() == "ADPCM":
            enc = "-f a "
        elif self.encoding.get() == "IMA_ADPCM":
            enc = "-f i "
        elif self.encoding.get() == "GMS":
            enc = "-f g "
        elif self.encoding.get() == "Floating-point":
            enc = "-f f "
        else: pass
        cmd += enc
        cmd += " -c %s " % self.channels.get()
        if self.swap.get():
            cmd += "-x "
        cmd += self.filename.get()
        print "Executing: %s" % cmd
        os.system(cmd)

if __name__ == "__main__":
    myApp = mainApp()
    myApp.mainloop()
