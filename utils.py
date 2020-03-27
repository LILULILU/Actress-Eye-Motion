from PyDAQmx import *
from PyDAQmx import Task
from PyDAQmx.DAQmxTypes import *

def init_task():

    amplifier = Task()
    amplifier.CreateAOVoltageChan("Dev1/ao0", "", -10.0, 10.0, DAQmx_Val_Volts, None)
    amplifier.StartTask()

    laser_sensor = Task()
    laser_sensor.CreateAIVoltageChan("Dev1/ai0", "", DAQmx_Val_Cfg_Default, -10.0, 10.0, DAQmx_Val_Volts, None)
    laser_sensor.CfgSampClkTiming("", 10000.0, DAQmx_Val_Rising, DAQmx_Val_ContSamps, 1000)
    laser_sensor.StartTask()

    return amplifier, laser_sensor
