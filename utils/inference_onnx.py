import onnxruntime
import numpy as np
import torch as T
import time
import os, sys

import psutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import ueEnv
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def cmvae_onnx():
    img = T.randn(1,2,144,256)
    ort_session = onnxruntime.InferenceSession("../onnx/cmvae.onnx", providers=['CPUExecutionProvider'])
    onnx_inputs= {ort_session.get_inputs()[0].name:to_numpy(img)}

    onnxruntime_outputs = ort_session.run(None, onnx_inputs)
    latent = onnxruntime_outputs[0]
    print(latent)

def d3qn_onnx():
    latent = T.randn(1,128)
    info = T.randn(1,7)
    input = T.cat((latent,info),dim=1)
    ort_session = onnxruntime.InferenceSession("../onnx/D3QN.onnx", providers=['CPUExecutionProvider'])
    onnx_inputs= {ort_session.get_inputs()[0].name:to_numpy(input)}

    onnxruntime_outputs = ort_session.run(None, onnx_inputs)
    action = onnxruntime_outputs[0]
    print(len(onnxruntime_outputs))

def ppo_onnx():
    latent = T.randn(1,128)
    info = T.randn(1,7)
    input = T.cat((latent,info),dim=1)
    ort_session = onnxruntime.InferenceSession("../onnx/PPO.onnx", providers=['CUDAExecutionProvider'])
    onnx_inputs= {ort_session.get_inputs()[0].name:to_numpy(input)}
    onnxruntime_outputs = ort_session.run(None, onnx_inputs)
    action = onnxruntime_outputs[0]
    print(action)
    
class pipline_D3QN():
    def __init__(self):
        session_options = onnxruntime.SessionOptions()
        session_options.log_severity_level = 1  # 設置日誌級別以獲取詳細日誌
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.cmvae_ort_session = onnxruntime.InferenceSession("../onnx/cmvae.onnx", providers=providers)
        self.d3qn_ort_session = onnxruntime.InferenceSession("../onnx/D3QN.onnx", providers=providers)

    def inference(self, images, info):
        cmvae_inputs = {self.cmvae_ort_session.get_inputs()[0].name:(images)}
        cmvae_outputs = self.cmvae_ort_session.run(None, cmvae_inputs)
        latent = cmvae_outputs[0]
        cat_input = np.concatenate((latent,info),axis=1)
        d3qn_inputs= {self.d3qn_ort_session.get_inputs()[0].name:(cat_input)}
        d3qn_outputs = self.d3qn_ort_session.run(None, d3qn_inputs)
        actions = d3qn_outputs[0]
        return np.argmax(actions,axis=1).item()

class pipline_PPO():
    def __init__(self):
        #session_options = onnxruntime.SessionOptions()
        #session_options.log_severity_level = 0  # verbose
        #session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ['CPUExecutionProvider'] # [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})] 
        self.cmvae_ort_session = onnxruntime.InferenceSession("../onnx/cmvae.onnx" ,providers=providers)
        self.ppo_ort_session = onnxruntime.InferenceSession("../onnx/PPO.onnx", providers=providers)
    
    def inference(self, images, info):
        io_binding = self.cmvae_ort_session.io_binding()
        io_binding.bind_cpu_input('images', images)
        io_binding.bind_output('latent')
        self.cmvae_ort_session.run_with_iobinding(io_binding)
        latent = io_binding.copy_outputs_to_cpu()[0]
        cat_input = np.concatenate((latent,info),axis=1)
        #cat_input = np.random.random((1,135)).astype(np.float32)
        io_binding = self.ppo_ort_session.io_binding()
        io_binding.bind_cpu_input('latent_with_info', cat_input)
        io_binding.bind_output('action')
        self.ppo_ort_session.run_with_iobinding(io_binding)
        action = io_binding.copy_outputs_to_cpu()[0]
        return action #latent
    
def testpipline():
    
    env = ueEnv()
    agent = pipline_PPO()
    done = False
    step = 0
    obs,info = env.reset_test()
    while not done:
        action = agent.inference(obs,info)
        nx_obs, nx_info, reward, done, collision, exceed, success = env.step(action,step)
        step += 1
        obs = nx_obs
        info = nx_info
    """
    print('RAM memory used:', psutil.virtual_memory()[2])
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    print('-----------------------------------------------------------------')

    for i in range(5):
        agent = pipline_PPO()
        print('RAM memory used:', psutil.virtual_memory()[2])
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
        images = np.random.random((1,2,144,256)).astype(np.float32)
        info = np.random.random((1,7)).astype(np.float32)
        start = time.time()
        action = agent.inference(images,info)
        end = time.time()
        print("[onnx] PPO+CMVAE inference take times : {}".format(end-start))

    """

testpipline()