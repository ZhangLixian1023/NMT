from rnn_frame import RNN_frame
from pprint import pprint
from transformer_frame import Transformer_frame
e = Transformer_frame()
#e = RNN_frame()
e.load_model(save=False)
#e.test([("我 爱 自然 语言 处理 。","i love natural language processing ."),
#        ("今天 天气 很 好 。","the weather is nice today .")])

loss = e.evaluate()
print(f'验证集 Loss: {loss:.3f}')
import time
strat=time.time()
e.test(e.test_pairs,show=False,strategy='beam')
end=time.time()
mins, secs = divmod(end - strat, 60)
print(f'Test 用时: {mins}m {secs:.0f}s')