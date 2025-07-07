import torch
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from tqdm import tqdm

from config import get_config, get_weights_path
from data_utils import load_dataset_and_tokenizer
from model import build_transformer
from dataset import BilingualDataset, causal_mask
import torchmetrics


def greedy_decode(
    model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device
):
    """
    Greedy decoding function to generate sequences from the model.
    """
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Precompute encoder output and reuse it for each decoding step
    encoder_output = model.encode(encoder_input, encoder_mask)

    # Initialize the decoder input with the SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)
        )

        # calculate output
        out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.linear(out[:, -1])  # last token
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1)
                .type_as(encoder_input)
                .fill_(next_word.item())
                .to(device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(
    model,
    val_dataset,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    print_message,
    global_step,
    writer,
    num_examples=2,
):
    model.eval()
    count = 0

    source_sentences = []
    target_sentences = []
    predicted_sentences = []

    console_width = 80

    with torch.no_grad():
        for batch in val_dataset:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (batch=1, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation."

            model_output = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len=max_len,
                device=device,
            )

            source_sentences.append(batch["src_text"][0])
            target_sentences.append(batch["tgt_text"][0])
            predicted_sentences.append(
                tokenizer_tgt.decode(model_output.detach().cpu().numpy())
            )

            print_message(f"-" * console_width)
            print_message(f"SOURCE: {source_sentences[-1]}")
            print_message(f"TARGET: {target_sentences[-1]}")
            print_message(f"PREDICTED: {predicted_sentences[-1]}")

            if count == num_examples:
                break

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted_sentences, target_sentences)
        writer.add_scalar("validation cer", cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted_sentences, target_sentences)
        writer.add_scalar("validation wer", wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted_sentences, target_sentences)
        writer.add_scalar("validation BLEU", bleu, global_step)
        writer.flush()


def get_model(config, vocab_src_size, vocab_tgt_size):
    """
    Create and return the model based on the configuration.
    """
    model = build_transformer(
        src_vocab=vocab_src_size,
        tgt_vocab=vocab_tgt_size,
        src_seq_len=config["seq_len"],
        tgt_seq_len=config["seq_len"],
        d_model=config["d_model"],
    )

    return model


def train_model(config):
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = (
        load_dataset_and_tokenizer(config)
    )

    model = get_model(
        config,
        vocab_src_size=tokenizer_src.get_vocab_size(),
        vocab_tgt_size=tokenizer_tgt.get_vocab_size(),
    ).to(device)

    # Tensorboard writer
    writer = SummaryWriter(config["experiment_name"])

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], eps=1e-9
    )

    init_epoch = 0
    global_step = 0
    if config["preload_weights"]:
        model_filename = get_weights_path(config, config["preload_weights"])
        print(f"Loading model weights from {model_filename}")
        state = torch.load(model_filename)
        init_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer"])
        global_step = state["global_step"]

    loss_function = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id("[PAD]"),
        label_smoothing=0.1,
    ).to(device)

    for epoch in range(init_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}"
        )

        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)  # (batch, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (batch, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(
                device
            )  # (batch, 1, seq_len, seq_len)
            label = batch["label"].to(device)  # (batch, seq_len)

            # run tensors through the model
            # (batch, seq_len) for encoder_input and decoder_input
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )

            # # (batch, seq_len, vocab_size) for decoder_output
            linear_output = model.linear(decoder_output)

            # (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
            loss = loss_function(
                linear_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )
            batch_iterator.set_postfix(loss=loss.item())

            # Log in TensorBoard
            writer.add_scalar("Train loss", loss.item(), global_step)

            # Backpropagation
            loss.backward()

            # update model parameters
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        run_validation(
            model,
            val_dataloader,
            tokenizer_src,
            tokenizer_tgt,
            max_len=config["seq_len"],
            device=device,
            print_message=lambda msg: batch_iterator.write(msg),
            global_step=global_step,
            writer=writer,
        )

        # Save the model weights at the end of every epoch
        model_filename = get_weights_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    config = get_config()
    print(f"Configuration: {config}")

    train_model(config)

    print("Training completed successfully.")
