import multiprocessing as mp
import time as t
from TestEmotionDetector import emotion_test
from GraphPlotting import graphPlot

p1 = mp.Process(target=emotion_test)
p2 = mp.Process(target=graphPlot)

if __name__ == '__main__':
    p1.start()
    t.sleep(5)
    p2.start()