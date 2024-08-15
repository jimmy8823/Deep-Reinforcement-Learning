import torch as T
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CM_VAE.CM_VAE_ver2 import CmVAE
from D3QN import D3QN
from stable_baselines3 import PPO
from typing import Tuple
from stable_baselines3.common.policies import BasePolicy

def export_cmvae():
    print(T.__version__)
    device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")
    cmvae_path = "D:/CODE/Python/AirSim/CM_VAE/mix_origin_depth/cmvae_50.pth"
    dummy_input = T.randn(1,2,144,256)
    cmvae = CmVAE(input_dim=2,semantic_dim=1,validation=True)
    #cmvae.load_state_dict(T.load(cmvae_path))
    cmvae.eval()
    input_names = [ "images" ]
    output_names = [ "latent" ]
    T.onnx.export(cmvae,
                 dummy_input,
                 "../onnx/CMVAE_test.onnx",
                 verbose=False,
                 input_names=input_names,
                 output_names=output_names,
                 export_params=True
                 )

def export_D3QN():
    device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")
    D3QN_path = "D:\\CODE\\Python\\AirSim\\D3QN_with_cmvae\\model\\D3QN_3000.pth"
    dummy_input = T.randn(1,135).to(device)
    d3qn = D3QN(input_dim=135, action_dim=7,lr=0, gamma=0.99, epsilon=0, 
                update_freq=5000, steps=0,path=D3QN_path, validation=True)
    d3qn.load_models()
    d3qn.q_net.eval()
    input_names = [ "latent_with_info"]
    output_names = [ "action" ]
    T.onnx.export(d3qn.q_net,
                 dummy_input,
                 "../onnx/D3QN.onnx",
                 verbose=False,
                 input_names=input_names,
                 output_names=output_names,
                 export_params=True,
                 )
    
class OnnxablePolicy(T.nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: T.Tensor) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:
        # NOTE: Preprocessing is included, but postprocessing
        # (clipping/inscaling actions) is not,
        # If needed, you also need to transpose the images so that they are channel first
        # use deterministic=False if you want to export the stochastic policy
        # policy() returns `actions, values, log_prob` for PPO
        return self.policy(observation, deterministic=False)
    
def export_PPO():
    device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")
    print(device)
    PPO_path = "D:\\CODE\\Python\\AirSim\\PPO\\PPO_CMVAE\\model\\quad_land_2_100000_steps"
    dummy_input = T.randn(1,135).to(device)
    model = PPO.load(PPO_path,device=device)
    onnxable_model = OnnxablePolicy(model.policy)
    input_names = [ "latent_with_info"]
    output_names = [ "action", "state" ]
    #output_names=output_names,
    T.onnx.export(onnxable_model,
                 dummy_input,
                 "../onnx/PPO.onnx",
                 verbose=False,
                 input_names=input_names,
                 output_names=output_names,
                 export_params=True,
                 )

export_PPO()