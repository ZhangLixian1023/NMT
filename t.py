from exp_frame import Exp_frame
from transformer_frame import transformer_frame
e = Exp_frame()
print("exp_setting:",e.exp_setting)
e.init_model()
e.train(10)