import argparse

RECONSTRUCTED_BASE_SENT = "The relation between \"{head_entity}\" and \"{tail_entity}\" in the context: \"{input_sent}\""


def get_params():
    # parse parameters
    parser = argparse.ArgumentParser(description="YxinMiracle LLM RE")
    parser.add_argument("--bert_model_name", type=str, default="bert-base-cased",
                        help="model name (e.g., bert-base-cased, roberta-base)")
    parser.add_argument("--input_size", type=int, default=768, help="bert output hidden")
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--saved_mode_name", type=str, default="model.pt")
    parser.add_argument("--train_log_file_name", type=str, default="train.log")
    parser.add_argument("--save_directory_name", type=str, default="save", help="data directory name")
    parser.add_argument("--log_directory_name", type=str, default="log", help="data directory name")
    parser.add_argument("--swa_warmup", type=int, default=10, help="swa_warmup")

    parser.add_argument("--seed", default=7777, type=int)
    parser.add_argument("--step", default=100, type=int)
    parser.add_argument("--batch_size", default=8, type=int,
                        help="number of samples in one training batch")
    parser.add_argument("--epoch", default=200, type=int,
                        help="number of training epoch")
    parser.add_argument("--hidden_size", default=300, type=int,
                        help="number of hidden neurons in the model")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropout rate for input word embedding")
    parser.add_argument("--steps", default=50, type=int,
                        help="show result for every 50 steps")
    parser.add_argument("--lr", default=0.0001, type=float,
                        help="initial learning rate")
    parser.add_argument("--clip", default=0.25, type=float,
                        help="grad norm clipping to avoid gradient explosion")
    params = parser.parse_args()
    return params


if __name__ == '__main__':
    params = get_params()
    print(params.bert_model_name)
