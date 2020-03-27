from PyDAQmx import *
from PyDAQmx import Task
from PyDAQmx.DAQmxTypes import *
import numpy as np
import time
import math
import sys
import csv
from apscheduler.schedulers.blocking import BlockingScheduler
import matplotlib.pyplot as plt
import csv
import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from DQN import DQN, ReplayMemory
import utils
from collections import namedtuple

random.seed(121)
# ========================================================================================
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
'''
input is: 
1)current position
#2)current velocity(current_position-previous_position)
3)previous voltage
4)desired position  #desired sequence
5)model predicted voltage
6)time
'''

smooth = 0.04

NUM_INPUT = 8
BASE_REWARD = 0.3
BATCH_SIZE = 100
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50
HIDDEN = 50
LAMBDA = 0.99
ACTION = np.concatenate((np.arange(-0.4, 0, 0.04), np.arange(0, 0.1, 0.01)))
NUM_ACTION = ACTION.shape[0]
epoch = 0
max_epoch = 100
init = True

if not os.path.exists('result/model'):
    os.mkdir('result/model')
    os.mkdir('result/test')
with open('result/config.txt', 'w') as f:
    f.write("base reward: {:f}\n".format(BASE_REWARD))
    f.write("batch size:: {:d}\n".format(BATCH_SIZE))
    f.write("gamma: {:f}\n".format(GAMMA))
    f.write("num input: {:d}\n".format(NUM_INPUT))
f.close()

policy_net = DQN(HIDDEN, NUM_ACTION, NUM_INPUT)
target_net = DQN(HIDDEN, NUM_ACTION, NUM_INPUT)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(8000)
lr_schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch, 0.0001)

def select_action(state, test=False):
    if test:
        with torch.no_grad():
            a = policy_net(state).max(1)[1].view(1, 1)
        return a
    else:
        global epoch
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * epoch / EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                a = policy_net(state).max(1)[1].view(1, 1)
            print('act according to model: %d\n' % a.squeeze())
            eps = 'model'
        else:
            a = torch.tensor([[random.randrange(NUM_ACTION)]], dtype=torch.long)
            print('act random: %d\n' % a.squeeze())
            eps = 'random'
        return a, eps


# ========================================================================================
def optimize_model():
    global init
    if init:
        init = False
        for param, param_target in zip(policy_net.parameters(), target_net.parameters()):
            param_target.data.copy_(param.data)
    for param, param_target in zip(policy_net.parameters(), target_net.parameters()):
        param_target.data.mul_(LAMBDA).add_((1 - LAMBDA), param.data)

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_states = torch.cat(batch.next_state)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    target_net.train()
    target_net(state_batch)
    target_net.eval()

    # Compute V(s_{t+1}) for all next states.
    next_state_values = target_net(next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# ========================================================================================
# read from csv
models = []
for t in range(10, 600):
    volt = []
    disp = []
    row_num = 12 + t
    for i in [1.5,2,2.5,3,3.5,3.8]:
        csvfile = open('data/%.1fkv_60s.csv' % i, 'r')
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        n = 0
        for row in reader:
            n = n + 1
            if n == row_num:
                volt.append(row[1])
                disp.append(row[2])
                break
    disp = np.asarray(disp, dtype='float')
    volt = np.asarray(volt, dtype='float')

    # fit curve and plot
    z = np.polyfit(disp, volt, 4)
    fitmodel = np.poly1d(z)
    models.append(fitmodel)

# ========================================================================================
# randomly get desired_trajectory
def rand_traj():

    desired_traj = np.zeros(10)

    y = np.random.random([50])*6
    rand_z = np.polyfit(np.linspace(1,50,50), y, 16)
    rand_p = np.poly1d(rand_z)
    desired = np.append(np.zeros(10), rand_p(np.linspace(5,49,450)))
    desired = np.clip(desired, 2, 4)

    desired_traj = np.append(desired_traj, desired)
    desired_traj = np.append(desired_traj, np.zeros(10))

    return desired_traj, desired

# ========================================================================================
def get_voltage(input,fit_model):

    volt_out = np.zeros(len(input))
    for s in range(len(input)):
        model = fit_model[s]
        volt_out[s] = model(input[s])
    volt_out = volt_out / 2

    #assert max(volt_out) < 2, "maximum voltage exceeds 4!"
    assert len(volt_out) == len(input), "control voltage and desired_trajectory not the same length!"

    return volt_out

# ========================================================================================
# initialize amplfier and laser
timeout = 0.1
sample_per_step = 1000
n = 480

test = []
test_actual = []

test_traj = np.zeros(10)
#test_desired = np.append(np.zeros(10), np.sin(0.2 * math.pi * np.linspace(-2.5, 37.5, 400)) + 3)
test_desired = np.append(np.zeros(10), np.sin((2/9) * math.pi * np.linspace(-9/4, 45-9/4, 450)) + 3)
test_traj = np.append(test_traj, test_desired)
test_traj = np.append(test_traj, np.zeros(10))
assert len(test_traj) == n
test.append(test_traj)
test_actual.append(test_desired)

test_traj = np.zeros(10)
#test_desired = np.append(np.zeros(10), np.sin(0.25 * math.pi * np.linspace(-2, 38, 400)) + 3)
test_desired = np.append(np.zeros(10), np.sin((2/15) * math.pi * np.linspace(-15/4, 45-15/4, 450)) + 3)
test_traj = np.append(test_traj, test_desired)
test_traj = np.append(test_traj, np.zeros(10))
assert len(test_traj) == n
test.append(test_traj)
test_actual.append(test_desired)

amp=4

test_traj = np.zeros(10)
test_desired = np.zeros(10)
for cycle in range(3):
    test_desired = np.append(test_desired, np.linspace(2,amp,75))
    test_desired = np.append(test_desired, np.linspace(amp,2,75))
test_traj = np.append(test_traj, test_desired)
test_traj = np.append(test_traj, np.zeros(10))
assert len(test_traj)==n
test.append(test_traj)
test_actual.append(test_desired)

test_traj = np.zeros(10)
test_desired = np.zeros(10)
for cycle in range(5):
    test_desired = np.append(test_desired, np.linspace(2,amp,45))
    test_desired = np.append(test_desired, np.linspace(amp,2,45))
test_traj = np.append(test_traj, test_desired)
test_traj = np.append(test_traj, np.zeros(10))
assert len(test_traj)==n
test.append(test_traj)
test_actual.append(test_desired)

test_traj = np.zeros(10)
test_desired = np.zeros(10)
for cycle in range(3):
    test_desired = np.append(test_desired, np.ones(75)*2)
    test_desired = np.append(test_desired, np.ones(75)*amp)
test_traj = np.append(test_traj, test_desired)
test_traj = np.append(test_traj, np.zeros(10))
assert len(test_traj)==n
test.append(test_traj)
test_actual.append(test_desired)

test_traj = np.zeros(10)
test_desired = np.zeros(10)
for cycle in range(5):
    test_desired = np.append(test_desired, np.ones(45)*2)
    test_desired = np.append(test_desired, np.ones(45)*amp)
test_traj = np.append(test_traj, test_desired)
test_traj = np.append(test_traj, np.zeros(10))
assert len(test_traj)==n
test.append(test_traj)
test_actual.append(test_desired)

data = np.zeros((sample_per_step,), dtype=np.float64)
while True:

    target_net.eval()
    policy_net.eval()

    laser = []
    read = int32(sample_per_step)
    i = 0
    amplifier, laser_sensor = utils.init_task()

    reward_list = []
    init_pos = 0
    input_voltage = 0
    state_list = []
    action_list = []
    voltage_list = []
    eps_list = []

    punish_list = []
    imp_list = []

    while True:
        desired_traj, desired = rand_traj()
        control_v = get_voltage(desired,models)
        if max(control_v)<1.8:
            break


    assert len(desired_traj) == n
    starttime = time.time()

    def iterate():
        global i
        global init_pos
        global input_voltage
        laser_sensor.ReadAnalogF64(sample_per_step, timeout, DAQmx_Val_GroupByChannel, data, sample_per_step,
                                   byref(read),
                                   None)
        dist = data.mean() * 20
        if i == 0:
            init_pos = dist
        dist = init_pos - dist

        reward = 0

        if i in range(11, n - 10):
            imp = abs(laser[i-1] - desired_traj[i-1])
            slope_desired = desired_traj[i]-desired_traj[i-1]
            slope_actual = dist - laser[i-1]
            punish = abs(slope_actual-slope_desired)

            imp = min(smooth/imp,1)
            reward = -abs(dist - desired_traj[i]) + BASE_REWARD - punish*imp
            reward_list.append(reward)

            punish_list.append(punish)
            imp_list.append(imp)
        if i in range(10, n - 11):
            input_time = (i - 10.0) / 100
            if i < 459:
                state = torch.Tensor([[dist, input_voltage, desired[i - 9], control_v[i - 9], input_time,
                                       desired[i - 4], desired[i + 1], desired[i-10]]])  # look ahead to next 0.5, 1
            else:
                state = torch.Tensor([[dist, input_voltage, desired[i - 9], control_v[i - 9], input_time,
                                       desired[i - 9], desired[i - 9], desired[i-10]]])
            action, act_eps = select_action(state)
            offset = ACTION[action]
            input_voltage = control_v[i - 9] + offset
            input_voltage = max(input_voltage, 0)
            eps_list.append(act_eps)
            action_list.append(action)
            state_list.append(state)
            voltage_list.append(input_voltage)
        else:
            input_voltage = 0
        amplifier.WriteAnalogScalarF64(1, 10.0, input_voltage, None)

        laser.append(dist)
        print('second %.1f  -- voltage %.1f -- distance %.1f -- reward %.2f\n'
              % ((i + 1) * timeout, input_voltage, dist, reward))
        i = i + 1
        if i == n:
            sched.shutdown(wait=False)


    sched = BlockingScheduler()
    sched.add_job(iterate, 'interval', seconds=0.1)
    sched.start()

    for i in range(len(state_list) - 1):
        memory.push(state_list[i], action_list[i], state_list[i + 1], torch.Tensor([reward_list[i]]))

    amplifier.StopTask()
    laser_sensor.StopTask()
    # ========================================================================================
    # plot and save readings
    laser = np.asarray(laser, dtype='float')
    plt.plot(range(n), laser)
    plt.plot(range(n), desired_traj)
    plt.legend(['laser output', 'desired output'])

    error = np.abs(desired - laser[10:-10]).sum() / len(desired)
    plt.title('error: %f' % error)
    plt.savefig('result/epoch_%d.png' % epoch)
    plt.close()

    with open('result/epoch_%d.csv' % epoch, 'w') as f:
        f.write('model_voltage,input_voltage,action,eps,current_disp,desired_disp,reward,punish,imp\n')
        for x in zip(state_list, action_list, reward_list, voltage_list, eps_list,punish_list,imp_list):
            f.write("{:f},{:f},{:d},{:s},{:f},{:f},{:f},{:f},{:f} \n".format(x[0][0][3], x[3], x[1].squeeze(), x[4], x[0][0][0],
                                                                   x[0][0][2], x[2],x[5],x[6]))

    if epoch >= 5:
        policy_net.train()
        print('Train model\n')
        for k in range(10*min(5,int((epoch + 1)**(1/2)))):
            for param_group in optimizer.param_groups:
                learning_rate = param_group['lr']
                print('learning rate: %.4f\n' %learning_rate)
            optimize_model()
        lr_schedular.step()
        policy_net.eval()
        torch.save(policy_net.state_dict(), './result/model/model_%d.pt' % epoch)
    # ========================================================================================
    # test
    if epoch >= 10 and epoch % 5 == 0:
        error_log = []
        test_dir = 'result/test/epoch%d'%epoch
        os.mkdir(test_dir)
        num_test = 1
        for traj_test, desired_test in zip(test,test_actual):
            test_volt = get_voltage(desired_test,models)
            for i in range(40, 0, -1):
                sys.stdout.write(str(i)+' ')
                sys.stdout.flush()
                time.sleep(1)
            amplifier, laser_sensor = utils.init_task()
            laser = []
            i = 0
            init_pos = 0
            input_voltage = 0
            state_list = []
            action_list = []
            voltage_list = []

            def iterate_test():
                global i
                global init_pos
                global input_voltage
                laser_sensor.ReadAnalogF64(sample_per_step, timeout, DAQmx_Val_GroupByChannel, data, sample_per_step,
                                           byref(read), None)
                dist = data.mean() * 20
                if i == 0:
                    init_pos = dist
                dist = init_pos - dist
                if i in range(10, n - 11):
                    input_time = (i - 10.0) / 100
                    if i < 459:
                        state = torch.Tensor([[dist, input_voltage, desired_test[i - 9], test_volt[i - 9], input_time,
                                               desired_test[i - 4], desired_test[i + 1], desired_test[i - 10]]])
                    else:
                        state = torch.Tensor([[dist, input_voltage, desired_test[i - 9], test_volt[i - 9], input_time,
                                               desired_test[i - 9], desired_test[i - 9], desired_test[i - 10]]])
                    action = select_action(state, True)
                    offset = ACTION[action]
                    if epoch == 0:
                        offset = 0
                    input_voltage = test_volt[i - 9] + offset
                    input_voltage = max(input_voltage, 0)
                    action_list.append(action)
                    state_list.append(state)
                    voltage_list.append(input_voltage)
                else:
                    input_voltage = 0
                amplifier.WriteAnalogScalarF64(1, 10.0, input_voltage, None)

                laser.append(dist)
                print('Test: second %.1f  -- voltage %.1f -- distance %.1f\n' % ((i + 1) * timeout, input_voltage, dist))
                i = i + 1
                if i == n:
                    sched.shutdown(wait=False)

            sched = BlockingScheduler()
            sched.add_job(iterate_test, 'interval', seconds=0.1)
            sched.start()

            amplifier.StopTask()
            laser_sensor.StopTask()

            laser = np.asarray(laser, dtype='float')
            plt.plot(range(n), laser)
            plt.plot(range(n), traj_test)
            plt.legend(['laser output', 'desired output'])

            error = np.abs(desired_test - laser[10:-10]).sum() / len(desired_test)
            plt.title('error: %f' % error)
            plt.savefig('%s/test_epoch%d_traj%d.png' % (test_dir,epoch,num_test))
            plt.close()
            error_log.append(error)

            with open('%s/test_epoch%d_traj%d.csv' % (test_dir,epoch,num_test), 'w') as f:
                f.write('model_voltage,input_voltage,action,current_disp,desired_disp\n')
                for x in zip(state_list, action_list, voltage_list):
                    f.write("{:f},{:f},{:d},{:f},{:f}\n".format(x[0][0][3], x[2], x[1].squeeze(), x[0][0][0], x[0][0][2]))
            num_test += 1
        with open('%s/error_log_epoch%d.csv' %(test_dir,epoch), 'w') as f:
            for x in error_log:
                f.write("{:f}\n".format(x))
            f.write("{:f}\n".format(sum(error_log)))
    # ========================================================================================
    epoch += 1
    for i in range(40, 0, -1):
        sys.stdout.write(str(i) + ' ')
        sys.stdout.flush()
        time.sleep(1)
    if epoch > max_epoch:
        break
