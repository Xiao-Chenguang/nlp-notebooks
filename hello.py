import torch
from torchinfo import summary
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator
from transformers import PreTrainedModel, XLNetModel


def print_model_info(xlnet: PreTrainedModel):
    print(f"This is the infomation for {xlnet.__class__.__name__}")

    # Define input dimensions (sequence length 128 and batch size 1, for example)
    input_ids = torch.randint(0, 32000, (1, 128))

    # Generate a summary
    summary(xlnet, input_data=(input_ids,))


def main():
    xlnet = XLNetModel.from_pretrained("xlnet-base-cased")
    print_model_info(xlnet)

    # load AG News from torchtext
    data_root = ".dataset"
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(
        map(tokenizer, AG_NEWS(root=data_root, split="train")), specials=["<unk>"]
    )
    vocab.set_default_index(vocab["<unk>"])

    print(f"Vocab size: {len(vocab)}")
    print(f"Vocab first 10: {vocab.get_itos()[:10]}")

    # Define input dimensions (sequence length 128 and batch size 1, for example)


if __name__ == "__main__":
    # redirect stdout to a file
    import sys

    sys.stdout = open("output.txt", "w")
    main()
    sys.stdout.close()
