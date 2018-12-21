"""
This program trains DDPG for adaptive bitrate streaming
"""

import os.path
import LiveStreamingEnv.env as env
import LiveStreamingEnv.load_trace as load_trace
import matplotlib.pyplot as plt
import time
import numpy as np
from DDPG import DDPG

# path setting
cwd = os.getcwd()
TRAIN_TRACES = cwd + os.sep + 'network_trace' + os.sep  # train trace path setting,
video_size_file = cwd + os.sep + 'video_trace' + os.sep + 'Fengtimo_2018_11_3' \
                  + os.sep + 'frame_trace_'  # video trace path setting
LogFile_Path = cwd + os.sep + 'log' + os.sep  # log file trace path setting
# load the trace
all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TRAIN_TRACES)
NN_MODEL = cwd + os.sep + 'abr_ddpg.ckpt'

# Debug Mode: if True, debug info can be found in logfile
# if False, no log but training is faster
DEBUG = False
DRAW = True

# random_seed
random_seed = 2
video_count = 0
FPS = 25
frame_time_len = 0.04

# initiate the environment
env_sim = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw,
                          random_seed=random_seed, logfile_path=LogFile_Path,
                          VIDEO_SIZE_FILE=video_size_file, Debug=DEBUG)

# actions
BIT_RATE = [500.0, 850.0, 1200.0, 1850.0]  # kbps
TARGET_BUFFER = [2.0, 3.0]  # seconds
# initial actions (index)
last_bit_rate = 0
bit_rate = 1
target_buffer = 0

# QOE parameters
SMOOTH_PENALTY = 0.02
REBUF_PENALTY = 1.5
LANTENCY_PENALTY = 0.005

# results to plot
idx = 0
id_list = []
buffer_record = []
throughput_record = []
reward_record = []

# DDPG hyper parameters
MAX_EPISODES = 5  # number of training with the same data set
LR_A = 1e-3  # learning rate for actor
LR_C = 1e-3  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
HYPER = [LR_A, LR_C, GAMMA, TAU, MEMORY_CAPACITY, BATCH_SIZE]

STATE_ALL = len(env_sim.get_video_frame(bit_rate, target_buffer))
S_LEN = 8  # take how many historic frames' state info
S_INFO = STATE_ALL-1  # take how many dimensions in the list returned by the simulator
S_DIM = S_INFO
A_DIM = len(BIT_RATE) * len(TARGET_BUFFER)  # combine two action spaces
a_bound = [2.]

# DDPG training
start = time.clock()  # capture the time used to run the program

ddpg = DDPG(A_DIM, S_DIM, a_bound, HYPER, NN_MODEL)
var = 3  # control exploration
episode_reward = []

for i in range(MAX_EPISODES + 1):
    ep_reward = 0
    num_frames = 0  # number of frames in the video trace
    if var >= 0.1:
        var *= .985  # decay the action randomness

    if DRAW:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        plt.ion()  # interactive plot, one more point per chunk

    # training in each episode until end_of_video is true
    while True:
        reward_frame = 0
        num_frames += 1

        # state info of current frame
        time, time_interval, send_data_size, chunk_len, rebuf, buffer_size, \
        play_time_len, end_delay, cdn_newest_id, download_id, cdn_has_frame, \
        decision_flag, buffer_flag, cdn_flag, end_of_video = env_sim.get_video_frame(
            bit_rate, target_buffer)
        # store the above state info into a list, except cdn_has_frame
        s_ = [time, time_interval, send_data_size, chunk_len, rebuf, buffer_size,
              play_time_len, end_delay, cdn_newest_id, download_id, decision_flag,
              buffer_flag, cdn_flag, end_of_video]  # remove cdn_has_frame
        s_ = np.array(s_)

        if num_frames == 1:
            reward_chunk = 0
            s = s_

        # compute reward of current frame
        if not cdn_flag:
            reward_frame = frame_time_len * float(
                BIT_RATE[bit_rate]) / 1000 - REBUF_PENALTY * rebuf - LANTENCY_PENALTY * end_delay
        else:
            reward_frame = -(REBUF_PENALTY * rebuf)
        if decision_flag or end_of_video:
            reward_frame += -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
            last_bit_rate = bit_rate

        reward_chunk += reward_frame

        # DDPG selects bitrate and target buffer before new chunk comes and then learn
        if decision_flag:
            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -2, 2)  # add randomness to action selection for exploration
            r = reward_chunk
            ddpg.store_transition(s, a, r / 10, s_)
            if ddpg.pointer > MEMORY_CAPACITY:
                ddpg.learn()

        s = s_

        if decision_flag:
            reward_chunk = 0

        ep_reward += reward_frame

        # results to plot
        if time_interval != 0:
            id_list.append(idx)
            idx += time_interval
            buffer_record.append(buffer_size)
            trace_idx = env_sim.get_trace_id()
            throughput_record.append(all_cooked_bw[trace_idx][int(idx / 0.5)] * 1000)
            reward_record.append(ep_reward)

        if decision_flag or end_of_video:
            # plot interested results
            if DRAW:
                ax1.plot(id_list, reward_record, 'g')
                ax1.set_ylabel("Reward")

                ax2.plot(id_list, buffer_record, 'r')
                ax2.set_ylabel("Buffer_size")

                ax3.plot(id_list, throughput_record, 'b')
                ax3.set_ylabel("Throughput")
                ax3.set_xlabel('Time')

                plt.draw()
                plt.pause(0.01)
                plt.show()

        if end_of_video:
            episode_reward.append(ep_reward)
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            break

end = time.clock()
print('Running time: %s Seconds' % (end - start))
