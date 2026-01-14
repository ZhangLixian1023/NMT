import argparse
from rnn_frame import RNN_frame
from transformer_frame import Transformer_frame
def main():
    parser = argparse.ArgumentParser(description="NMT experiment")
    parser.add_argument("--norm", type=str, default='rmsnorm',help="norm type") # rmsnorm / layernorm
    parser.add_argument("-r","--relative",action="store_true",default=False,help="relative or absolute position embedding")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    args = parser.parse_args()

    print(f"learning rate = {args.lr}")
    print(f"norm type = {args.norm}")
    print(f"relative position = {args.relative}")

    e = Transformer_frame()
    e.exp_setting["learning_rate"] = args.lr
    e.model_params["relative_position"] = args.relative
    e.model_params["norm"] = args.norm

    e.init_model(save=True)

    e.train(n_epochs=15)

if __name__ == "__main__":
    main()