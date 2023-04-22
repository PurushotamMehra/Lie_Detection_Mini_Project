import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

def graphPlot():
    def plotting(i):

        df = pd.read_csv("EmotionsDetected.csv")
        time = df['Time']
        emotion = df['Emotions']

        ax.cla()

        ax.plot(emotion, label="Emotion", lw =2, color = 'Red')
        ax.set_ylim(-4, 4)
        ax.set_yticks((-2, -1, 0, 1, 2), ('Truth', '', 'Neutral', '', 'Lies'), color = 'Green')
        plt.legend(loc = 'upper left')
        # plt.ylabel("Emotions")
        plt.xlabel("Time", color = 'Blue')
        plt.tight_layout()


    fig = plt.figure(figsize=(10,5), facecolor="#DEDEDE")
    ax = plt.axes()
    ax.set_facecolor("#DEDEDE")

    ani = FuncAnimation(fig, plotting, interval = 500)
    plt.show() 

