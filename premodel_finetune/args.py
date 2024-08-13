import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--pre_model", type=str, help="pre_model", default = "bert-base-chinese")
parser.add_argument("--train_file", type=str, help="train", default = "./data/afqmc_public/train.json")
parser.add_argument("--valid_file", type=str, help="valid", default = "./data/afqmc_public/dev.json")
parser.add_argument("--lr", type=float, help="learning_rate", default = 1e-5)
parser.add_argument("--epoch_num", type=int, help="epoch_num", default = 3)
parser.add_argument("--batch_size", type=int, help="batch_size", default = 4)

# parser.add_argument("--device", type=int, help="epoch_num", default = 3)
args = parser.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'