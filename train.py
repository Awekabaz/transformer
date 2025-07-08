import torch
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from tqdm import tqdm

from config import get_config, get_weights_path
from data_utils import load_dataset_and_tokenizer
from model import build_transformer
from dataset import BilingualDataset, causal_mask
from utils import configure_reproducibility
import torchmetrics
from torchmetrics.text import BLEUScore


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


def calculate_validation_loss(model, val_dataset, tokenizer_tgt, device):
    """Calculate validation loss over the entire validation set."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id("[PAD]"),
        reduction='sum'
    )
    
    with torch.no_grad():
        for batch in val_dataset:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)
            
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            linear_output = model.linear(decoder_output)
            
            loss = loss_fn(linear_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            # Count non-padding tokens
            non_pad_tokens = (label != tokenizer_tgt.token_to_id("[PAD]")).sum().item()
            
            total_loss += loss.item()
            total_tokens += non_pad_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity.item()


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
    
    # Full validation set evaluation
    all_source_sentences = []
    all_target_sentences = []
    all_predicted_sentences = []
    
    console_width = 80
    
    print_message(f"Running full validation set evaluation...")
    
    with torch.no_grad():
        for batch in tqdm(val_dataset, desc="Validating", leave=False):
            encoder_input = batch["encoder_input"].to(device)
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
            
            all_source_sentences.append(batch["src_text"][0])
            all_target_sentences.append(batch["tgt_text"][0])
            all_predicted_sentences.append(
                tokenizer_tgt.decode(model_output.detach().cpu().numpy())
            )
    
    # Calculate validation loss and perplexity
    val_loss, val_perplexity = calculate_validation_loss(model, val_dataset, tokenizer_tgt, device)
    
    # Show a few examples
    print_message(f"\nValidation Examples:")
    print_message(f"-" * console_width)
    for i in range(min(num_examples, len(all_predicted_sentences))):
        print_message(f"SOURCE: {all_source_sentences[i]}")
        print_message(f"TARGET: {all_target_sentences[i]}")
        print_message(f"PREDICTED: {all_predicted_sentences[i]}")
        print_message(f"-" * console_width)
    
    if writer:
        # Validation loss and perplexity
        writer.add_scalar("validation/loss", val_loss, global_step)
        writer.add_scalar("validation/perplexity", val_perplexity, global_step)
        
        # Character Error Rate
        cer_metric = torchmetrics.CharErrorRate()
        cer = cer_metric(all_predicted_sentences, all_target_sentences)
        writer.add_scalar("validation/cer", cer, global_step)
        
        # Word Error Rate
        wer_metric = torchmetrics.WordErrorRate()
        wer = wer_metric(all_predicted_sentences, all_target_sentences)
        writer.add_scalar("validation/wer", wer, global_step)
        
        # BLEU Score
        bleu_metric = BLEUScore()
        bleu = bleu_metric(all_predicted_sentences, all_target_sentences)
        writer.add_scalar("validation/bleu", bleu, global_step)
        
        
        # Length statistics
        avg_src_len = sum(len(s.split()) for s in all_source_sentences) / len(all_source_sentences)
        avg_tgt_len = sum(len(s.split()) for s in all_target_sentences) / len(all_target_sentences)
        avg_pred_len = sum(len(s.split()) for s in all_predicted_sentences) / len(all_predicted_sentences)
        
        writer.add_scalar("validation/avg_source_length", avg_src_len, global_step)
        writer.add_scalar("validation/avg_target_length", avg_tgt_len, global_step)
        writer.add_scalar("validation/avg_prediction_length", avg_pred_len, global_step)
        
        # Dataset size
        writer.add_scalar("validation/dataset_size", len(all_predicted_sentences), global_step)
        
        writer.flush()
        
        # Print validation summary
        print_message(f"\nValidation Results (Step {global_step}):")
        print_message(f"  Dataset size: {len(all_predicted_sentences)}")
        print_message(f"  Validation Loss: {val_loss:.4f}")
        print_message(f"  Perplexity: {val_perplexity:.4f}")
        print_message(f"  BLEU Score: {bleu:.4f}")
        print_message(f"  Character Error Rate: {cer:.4f}")
        print_message(f"  Word Error Rate: {wer:.4f}")
        print_message(f"  Avg Source Length: {avg_src_len:.1f}")
        print_message(f"  Avg Target Length: {avg_tgt_len:.1f}")
        print_message(f"  Avg Prediction Length: {avg_pred_len:.1f}")
        print_message(f"-" * console_width)


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
    # Configure reproducibility
    configure_reproducibility(config)
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = (
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
        model.load_state_dict(state["model_state_dict"])
        init_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
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
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], global_step)

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
