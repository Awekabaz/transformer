import torch
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from config import get_config, get_weights_path
from data_utils import load_dataset_and_tokenizer
from model import build_transformer
from train import greedy_decode, get_model
from utils import configure_reproducibility

import torchmetrics
from torchmetrics.text import BLEUScore
from torchmetrics.text.bleu import BLEUScore as BLEUScoreOriginal
from torchmetrics import CharErrorRate, WordErrorRate


class ComprehensiveEvaluator:
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.metrics = {
            'bleu': BLEUScore(),
            'cer': CharErrorRate(),
            'wer': WordErrorRate(),
        }
        
    def calculate_perplexity(self, model, dataloader, tokenizer_tgt):
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=tokenizer_tgt.token_to_id("[PAD]"),
            reduction='sum'
        )
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Calculating perplexity"):
                encoder_input = batch["encoder_input"].to(self.device)
                decoder_input = batch["decoder_input"].to(self.device)
                encoder_mask = batch["encoder_mask"].to(self.device)
                decoder_mask = batch["decoder_mask"].to(self.device)
                label = batch["label"].to(self.device)
                
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                linear_output = model.linear(decoder_output)
                
                loss = loss_fn(linear_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                
                # Count non-padding tokens
                non_pad_tokens = (label != tokenizer_tgt.token_to_id("[PAD]")).sum().item()
                
                total_loss += loss.item()
                total_tokens += non_pad_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))
        return perplexity.item()
    
    def evaluate_dataset(self, model, dataloader, tokenizer_src, tokenizer_tgt, dataset_name: str) -> Dict:
        model.eval()
        
        source_texts = []
        target_texts = []
        predicted_texts = []
        
        print(f"\nEvaluating {dataset_name} set...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Generating {dataset_name} predictions"):
                encoder_input = batch["encoder_input"].to(self.device)
                encoder_mask = batch["encoder_mask"].to(self.device)
                
                model_output = greedy_decode(
                    model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt,
                    max_len=self.config["seq_len"], device=self.device
                )
                
                source_texts.append(batch["src_text"][0])
                target_texts.append(batch["tgt_text"][0])
                predicted_texts.append(tokenizer_tgt.decode(model_output.detach().cpu().numpy()))
        
        # Calculate metrics
        results = {}
        
        # BLEU Score
        bleu_score = self.metrics['bleu'](predicted_texts, target_texts)
        results['bleu'] = bleu_score.item()
        
        
        # Character Error Rate
        cer = self.metrics['cer'](predicted_texts, target_texts)
        results['cer'] = cer.item()
        
        # Word Error Rate
        wer = self.metrics['wer'](predicted_texts, target_texts)
        results['wer'] = wer.item()
        
        # Perplexity
        perplexity = self.calculate_perplexity(model, dataloader, tokenizer_tgt)
        results['perplexity'] = perplexity
        
        # Additional statistics
        results['num_samples'] = len(predicted_texts)
        results['avg_src_length'] = np.mean([len(text.split()) for text in source_texts])
        results['avg_tgt_length'] = np.mean([len(text.split()) for text in target_texts])
        results['avg_pred_length'] = np.mean([len(text.split()) for text in predicted_texts])
        
        return results, source_texts, target_texts, predicted_texts
    
    def save_results(self, results: Dict, output_dir: Path):
        output_dir.mkdir(exist_ok=True)
        
        # Save metrics as JSON
        metrics_file = output_dir / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {metrics_file}")
    
    def save_predictions(self, source_texts: List[str], target_texts: List[str], 
                        predicted_texts: List[str], output_dir: Path, dataset_name: str, max_samples: int = 100):
        output_dir.mkdir(exist_ok=True)
        
        # Limit the number of samples to save
        num_to_save = min(max_samples, len(source_texts))
        
        predictions_file = output_dir / f"{dataset_name}_predictions.jsonl"
        with open(predictions_file, 'w') as f:
            for i in range(num_to_save):
                sample = {
                    "source": source_texts[i],
                    "target": target_texts[i],
                    "prediction": predicted_texts[i]
                }
                f.write(json.dumps(sample) + "\n")
        
        print(f"Predictions saved to {predictions_file} ({num_to_save}/{len(source_texts)} samples)")


def load_model_from_checkpoint(config: Dict, checkpoint_path: str, tokenizer_src, tokenizer_tgt, device: torch.device):
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    print(f"Loading model from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    
    return model


def evaluate_model(checkpoint_path: str = None, output_dir: str = "evaluation_results", max_predictions: int = 100):
    config = get_config()
    
    # Configure reproducibility for evaluation
    configure_reproducibility(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_dataloader, val_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = load_dataset_and_tokenizer(config)
    
    # Load model
    if checkpoint_path is None:
        # Find the latest checkpoint
        weights_dir = Path(config["model_folder"])
        if weights_dir.exists():
            checkpoints = list(weights_dir.glob("*.pt"))
            if checkpoints:
                checkpoint_path = max(checkpoints, key=lambda x: x.stat().st_mtime)
            else:
                raise FileNotFoundError("No model checkpoints found")
        else:
            raise FileNotFoundError("Model folder not found")
    
    model = load_model_from_checkpoint(config, checkpoint_path, tokenizer_src, tokenizer_tgt, device)
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(config, device)
    
    # Evaluate on all datasets
    results = {}
    output_path = Path(output_dir)
    
    datasets = [
        ("validation", val_dataloader),
        ("test", test_dataloader)
    ]
    
    all_predictions = {}
    
    for dataset_name, dataloader in datasets:
        dataset_results, src_texts, tgt_texts, pred_texts = evaluator.evaluate_dataset(
            model, dataloader, tokenizer_src, tokenizer_tgt, dataset_name
        )
        
        results[dataset_name] = dataset_results
        all_predictions[dataset_name] = (src_texts, tgt_texts, pred_texts)
        
        # Save predictions (limited to max_predictions)
        evaluator.save_predictions(src_texts, tgt_texts, pred_texts, output_path, dataset_name, max_predictions)
    
    # Save all results
    evaluator.save_results(results, output_path)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name.upper()} SET:")
        print(f"  Samples: {dataset_results['num_samples']}")
        print(f"  BLEU Score: {dataset_results['bleu']:.4f}")
        print(f"  Character Error Rate: {dataset_results['cer']:.4f}")
        print(f"  Word Error Rate: {dataset_results['wer']:.4f}")
        print(f"  Perplexity: {dataset_results['perplexity']:.4f}")
        print(f"  Avg Source Length: {dataset_results['avg_src_length']:.1f}")
        print(f"  Avg Target Length: {dataset_results['avg_tgt_length']:.1f}")
        print(f"  Avg Prediction Length: {dataset_results['avg_pred_length']:.1f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate transformer model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--max-predictions", type=int, default=100, help="Maximum number of predictions to save per dataset")
    
    args = parser.parse_args()
    
    results = evaluate_model(args.checkpoint, args.output, args.max_predictions)