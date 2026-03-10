import diplib as dip
import numpy as np
import logging


def getSampleAxis(mask):
    '''Get the principal axis of the binary sample mask (longest axis of the sample))'''
    #A = dip.Image(img)
    dipimg = dip.Image()
    dipB = dip.Label(dip.Image(mask), minSize=1000)
    m = dip.MeasurementTool.Measure(dipB, dipimg, ["MajorAxes"])
    I_l = m["MajorAxes"][1][-3:]    # inertia axis for length of sample has lowes EV

    return I_l
