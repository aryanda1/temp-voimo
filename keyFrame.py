import wave
import struct
import numpy as np
import subprocess

def keyFrames(path,fps=12,fn="1+pow(x,4)"):
    wav_name='wave.wav'
    #If input audio not in wav convert in .wav
    if(path.split('\\')[-1].split('.')[-1])!='wav':
        cmd = ['ffmpeg',
              '-i',path,
              wav_name]
        process = subprocess.Popen(cmd,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout,stderr = process.communicate()
        path=wav_name
    wv = wave.open(path,'rb')
    nFrames = wv.getnframes()#Read the number of frames
    no_ch = wv.getnchannels()# Read the number of channels
    frameRate = wv.getframerate() # Read the sampling frequency
#     samp_width = wv.getsampwidth()
    dur = nFrames/frameRate # duration of audio clip

    a = wv.readframes(nFrames) # Read Frames*samp_width*no_ch
    wv.close()

    unpstr = '<{0}h'.format(nFrames*no_ch)
#     print(unpstr)
    x = np.absolute(np.array(struct.unpack(unpstr,a)))#convert the byte string into a numpy array of ints with absolute value
    x = x.reshape((no_ch,nFrames),order='F') #Splitting the channels by increasing the dimension of 1d to no of channels
    x = np.add.reduce(x)/no_ch # Averaging across all channels

    samples = dur*fps # no of samples we want
    blockSize = int(len(x)/samples) #no of frames in each block
    filterData = [] #stores the amplitude value for each samples frame
    for i in range(0,len(x),blockSize):
        chunk = x[i:i+blockSize]
        summ = sum(chunk)
        filterData.append(summ/len(chunk)) #Averaging over all the blockSize

    maxx = max(filterData)
    filterData = [eval(fn,{"x":data/maxx}) for data in filterData] #Evaluating by the fn and nomalizing filterData using max value

    s = ''
    for idx,data in enumerate(filterData):
        s+='{idx}: ({0:.2f}), '.format(data,idx=idx)
    s = s[:-2]
    maxFrames = s.split(',')[-1].split(':')[0]
    return s,int(maxFrames)
    # file = open("keyframes.txt", "w")
    # a = file.write(s)
    # file.close()

# def getkey():
#   file=open("keyframes.txt","r")
#   s = file.read()
#   maxFrames = s[-11:-8]
#   return s,int(maxFrames)