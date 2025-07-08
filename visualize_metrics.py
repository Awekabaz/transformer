import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple
import tensorboard.backend.event_processing.event_accumulator as ea


def extract_tensorboard_data(log_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """Extract metrics from TensorBoard logs."""
    event_acc = ea.EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get available scalar tags
    scalar_tags = event_acc.Tags()['scalars']
    
    data = {}
    for tag in scalar_tags:
        scalar_events = event_acc.Scalars(tag)
        data[tag] = [(event.step, event.value) for event in scalar_events]
    
    return data


def create_training_plots(tb_data: Dict, output_dir: Path):
    """Create comprehensive training visualization plots."""
    output_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Training Metrics Dashboard', fontsize=16)
    
    # Plot 1: Training Loss
    train_loss_key = 'train/loss' if 'train/loss' in tb_data else 'Train loss'
    if train_loss_key in tb_data:
        steps, losses = zip(*tb_data[train_loss_key])
        axes[0, 0].plot(steps, losses, label='Training Loss', linewidth=2)
        axes[0, 0].set_title('Training Loss Over Time')
        axes[0, 0].set_xlabel('Global Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    
    # Plot 2: Validation Loss
    val_loss_key = 'validation/loss' if 'validation/loss' in tb_data else None
    if val_loss_key and val_loss_key in tb_data:
        steps, losses = zip(*tb_data[val_loss_key])
        epochs = np.arange(1, len(losses) + 1)
        axes[0, 1].plot(epochs, losses, color='red', linewidth=2, marker='o')
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Training vs Validation Loss Comparison
    train_loss_key = 'train/loss' if 'train/loss' in tb_data else 'Train loss'
    val_loss_key = 'validation/loss' if 'validation/loss' in tb_data else None
    
    if train_loss_key in tb_data and val_loss_key and val_loss_key in tb_data:
        # Plot training loss (sampled to match validation frequency)
        train_steps, train_losses = zip(*tb_data[train_loss_key])
        val_steps, val_losses = zip(*tb_data[val_loss_key])
        val_epochs = np.arange(1, len(val_losses) + 1)
        
        # Sample training loss at validation epochs (approximate)
        if len(train_losses) >= len(val_losses):
            sample_indices = np.linspace(0, len(train_losses)-1, len(val_losses), dtype=int)
            sampled_train_losses = [train_losses[i] for i in sample_indices]
            
            axes[0, 2].plot(val_epochs, sampled_train_losses, label='Training Loss', 
                          linewidth=2, marker='s', color='blue', alpha=0.7)
            axes[0, 2].plot(val_epochs, val_losses, label='Validation Loss', 
                          linewidth=2, marker='o', color='red')
            
            axes[0, 2].set_title('Training vs Validation Loss')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].legend()
    elif val_loss_key and val_loss_key in tb_data:
        # Fallback to just validation loss if training loss sampling fails
        steps, val_losses = zip(*tb_data[val_loss_key])
        epochs = np.arange(1, len(val_losses) + 1)
        axes[0, 2].plot(epochs, val_losses, color='red', linewidth=2, marker='o')
        axes[0, 2].set_title('Validation Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: BLEU Score Progression
    bleu_key = 'validation/bleu' if 'validation/bleu' in tb_data else 'validation BLEU'
    if bleu_key in tb_data:
        steps, bleu_scores = zip(*tb_data[bleu_key])
        epochs = np.arange(1, len(bleu_scores) + 1)
        axes[1, 0].plot(epochs, bleu_scores, color='green', linewidth=3, marker='o')
        axes[1, 0].fill_between(epochs, bleu_scores, alpha=0.3, color='green')
        axes[1, 0].set_title('BLEU Score Progression')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('BLEU Score')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Translation Quality Metrics
    quality_metrics = ['validation/bleu', 'validation/cer', 'validation/wer']
    # Fallback to old naming convention
    if not any(metric in tb_data for metric in quality_metrics):
        quality_metrics = ['validation BLEU', 'validation cer', 'validation wer']
    
    for metric in quality_metrics:
        if metric in tb_data:
            steps, values = zip(*tb_data[metric])
            epochs = np.arange(1, len(values) + 1)
            clean_name = metric.replace('validation/', '').replace('validation ', '').upper()
            
            # Use different colors and markers for each metric
            if 'bleu' in metric.lower():
                axes[1, 1].plot(epochs, values, label=clean_name, linewidth=3, 
                              marker='o', color='green', markersize=8)
            elif 'cer' in metric.lower():
                axes[1, 1].plot(epochs, values, label=clean_name, linewidth=2, 
                              marker='s', color='orange', markersize=6)
            elif 'wer' in metric.lower():
                axes[1, 1].plot(epochs, values, label=clean_name, linewidth=2, 
                              marker='^', color='purple', markersize=6)
    
    axes[1, 1].set_title('Translation Quality Metrics')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score / Error Rate')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Plot 6: Perplexity
    perplexity_key = 'validation/perplexity' if 'validation/perplexity' in tb_data else None
    if perplexity_key and perplexity_key in tb_data:
        steps, perplexity_values = zip(*tb_data[perplexity_key])
        epochs = np.arange(1, len(perplexity_values) + 1)
        axes[1, 2].plot(epochs, perplexity_values, color='orange', linewidth=2, marker='o')
        axes[1, 2].set_title('Validation Perplexity')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Perplexity')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_yscale('log')  # Log scale for perplexity
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual detailed plots
    create_individual_plots(tb_data, output_dir)


def create_individual_plots(tb_data: Dict, output_dir: Path):
    """Create individual detailed plots for each metric."""
    
    # Training Loss with smoothing
    if 'Train loss' in tb_data:
        plt.figure(figsize=(12, 6))
        steps, losses = zip(*tb_data['Train loss'])
        
        plt.subplot(1, 2, 1)
        plt.plot(steps, losses, alpha=0.7, label='Raw Loss')
        
        # Add smoothed line
        if len(losses) > 10:
            smooth_losses = pd.Series(losses).rolling(window=min(100, len(losses)//10)).mean()
            plt.plot(steps, smooth_losses, linewidth=2, color='red', label='Smoothed Loss')
        
        plt.title('Training Loss')
        plt.xlabel('Global Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Log scale version
        plt.subplot(1, 2, 2)
        plt.plot(steps, losses, alpha=0.7, label='Raw Loss')
        if len(losses) > 10:
            plt.plot(steps, smooth_losses, linewidth=2, color='red', label='Smoothed Loss')
        plt.title('Training Loss (Log Scale)')
        plt.xlabel('Global Step')
        plt.ylabel('Loss (log scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_loss_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Validation metrics detailed plots
    val_metrics = ['validation/loss', 'validation/perplexity', 'validation/cer', 'validation/wer', 'validation/bleu']
    # Fallback to old naming convention
    old_val_metrics = ['validation cer', 'validation wer', 'validation BLEU']
    if not any(metric in tb_data for metric in val_metrics[:2]):  # Check new metrics
        val_metrics = old_val_metrics
    
    available_metrics = [metric for metric in val_metrics if metric in tb_data]
    
    if available_metrics:
        plt.figure(figsize=(15, 10))
        
        for i, metric in enumerate(available_metrics):
            if i >= 6:  # Limit to 6 subplots
                break
                
            steps, values = zip(*tb_data[metric])
            epochs = np.arange(1, len(values) + 1)
            
            plt.subplot(2, 3, i+1)
            plt.plot(epochs, values, linewidth=2, marker='o', markersize=8)
            
            clean_name = metric.replace('validation/', '').replace('validation ', '')
            plt.title(f'{clean_name.upper()} Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel(clean_name.upper())
            plt.grid(True, alpha=0.3)
            
            # Use log scale for perplexity
            if 'perplexity' in metric.lower():
                plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'validation_metrics_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_evaluation_comparison(results_file: str, output_dir: Path):
    """Create comparison plots from evaluation results."""
    if not Path(results_file).exists():
        print(f"Results file {results_file} not found")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create comparison bar chart
    metrics = ['bleu', 'cer', 'wer']
    datasets = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Evaluation Comparison', fontsize=16)
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        
        values = [results[dataset][metric] for dataset in datasets]
        bars = ax.bar(datasets, values, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_title(f'{metric.upper()} Comparison')
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_report(tb_data: Dict, results_file: str, output_dir: Path):
    """Create a comprehensive summary report."""
    output_dir.mkdir(exist_ok=True)
    
    report_file = output_dir / 'training_report.md'
    
    with open(report_file, 'w') as f:
        f.write("# Training and Evaluation Report\n\n")
        
        # Training summary
        f.write("## Training Summary\n\n")
        if 'Train loss' in tb_data:
            final_loss = tb_data['Train loss'][-1][1]
            f.write(f"- Final Training Loss: {final_loss:.4f}\n")
            f.write(f"- Total Training Steps: {tb_data['Train loss'][-1][0]}\n")
        
        # Validation summary
        f.write("\n## Validation Summary\n\n")
        val_metrics = ['validation cer', 'validation wer', 'validation BLEU']
        for metric in val_metrics:
            if metric in tb_data:
                final_value = tb_data[metric][-1][1]
                f.write(f"- Final {metric.replace('validation ', '').upper()}: {final_value:.4f}\n")
        
        # Evaluation results
        if Path(results_file).exists():
            f.write("\n## Final Evaluation Results\n\n")
            with open(results_file, 'r') as res_f:
                results = json.load(res_f)
            
            for dataset, metrics in results.items():
                f.write(f"### {dataset.capitalize()} Set\n\n")
                f.write(f"- Samples: {metrics['num_samples']}\n")
                f.write(f"- BLEU Score: {metrics['bleu']:.4f}\n")
                f.write(f"- Character Error Rate: {metrics['cer']:.4f}\n")
                f.write(f"- Word Error Rate: {metrics['wer']:.4f}\n")
                f.write(f"- Perplexity: {metrics['perplexity']:.4f}\n\n")
        
        f.write("## Generated Plots\n\n")
        f.write("- `training_dashboard.png`: Overview of all training metrics\n")
        f.write("- `training_loss_detailed.png`: Detailed training loss analysis\n")
        f.write("- `validation_metrics_detailed.png`: Detailed validation metrics\n")
        f.write("- `evaluation_comparison.png`: Final evaluation comparison\n")
    
    print(f"Summary report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics and evaluation results")
    parser.add_argument("--tb_logs", type=str, default="runs/transformer", 
                       help="TensorBoard logs directory")
    parser.add_argument("--results", type=str, default="evaluation_results/evaluation_metrics.json",
                       help="Evaluation results JSON file")
    parser.add_argument("--output", type=str, default="visualizations",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Extract TensorBoard data
    if Path(args.tb_logs).exists():
        print("Extracting TensorBoard data...")
        tb_data = extract_tensorboard_data(args.tb_logs)
        
        # Create training plots
        print("Creating training plots...")
        create_training_plots(tb_data, output_dir)
        
        # Create evaluation comparison
        if Path(args.results).exists():
            print("Creating evaluation comparison...")
            create_evaluation_comparison(args.results, output_dir)
        
        # Create summary report
        print("Creating summary report...")
        create_summary_report(tb_data, args.results, output_dir)
        
        print(f"All visualizations saved to {output_dir}")
        
    else:
        print(f"TensorBoard logs directory {args.tb_logs} not found")


if __name__ == "__main__":
    main()