# Transformer for Machine Translation

A complete PyTorch implementation of the Transformer architecture for sequence-to-sequence machine translation (English to Italian), following the "Attention is All You Need" paper.

- Original paper: "Attention is All You Need"
- Detailed explanations in code comments
- Visualization of attention patterns available in evaluation

## Architecture

### Model Structure
- **Encoder-Decoder Architecture**: 6-layer encoder and decoder stacks
- **Multi-Head Attention**: 8 attention heads with 512-dimensional model (d_model)
- **Feed-Forward Networks**: 2048-dimensional inner layer
- **Positional Encoding**: Sinusoidal positional embeddings
- **Layer Normalization**: Applied before each sub-layer (Pre-LN)
- **Residual Connections**: Around each sub-layer

### Key Components
- **`model.py`**: Complete transformer implementation with encoder/decoder stacks
- **`train.py`**: Training loop with comprehensive validation and logging
- **`dataset.py`**: BilingualDataset class with causal masking
- **`data_utils.py`**: Dataset loading and tokenizer management
- **`config.py`**: Centralized configuration management
- **`evaluate.py`**: Comprehensive evaluation framework
- **`visualize_metrics.py`**: Training visualization and plotting tools

### Tokenization
- **WordLevel tokenization** with special tokens: `[UNK]`, `[PAD]`, `[SOS]`, `[EOS]`
- **Vocabulary**: Built from Helsinki-NLP/opus_books dataset
- **Sequence Length**: 350 tokens (configurable)

## Dataset

- **Source**: Helsinki-NLP/opus_books (English-Italian translation)
- **Split**: 80% training, 10% validation, 10% test
- **Preprocessing**: Sequences padded/truncated to 350 tokens
- **Batch Size**: Varies by GPU memory (see GPU recommendations below), 1 for validation/test
   - **Why Validation Batch Size = 1**
      - The `greedy_decode()` function is designed for single sequences

## Quick Start

### Training
```bash
# Start training from scratch
python train.py

# Resume from checkpoint (edit config.py first)
# Set "preload_weights": "04" for epoch 4
python train.py
```

### Evaluation
```bash
# Evaluate latest checkpoint
python evaluate.py

# Evaluate specific checkpoint
python evaluate.py --checkpoint weights/transformer_model_04.pt --output evaluation_results

# Limit saved predictions (default: 100 per dataset)
python evaluate.py --max-predictions 50
```

### Visualization
```bash
# Generate all plots and reports
python visualize_metrics.py

# Custom paths
python visualize_metrics.py --tb_logs runs/transformer --results evaluation_results/evaluation_metrics.json --output visualizations
```

### Monitor Training
```bash
# Launch TensorBoard
tensorboard --logdir runs/transformer
```

## Training Configuration

### Default Hyperparameters
```python
{
    "batch_size": 12,  # Adjust based on GPU memory
    "num_epochs": 3,
    "learning_rate": 1e-4,
    "seq_len": 350,
    "d_model": 512,
    "train_size": 0.8,
    "val_size": 0.1,
    "language_src": "en",
    "language_tgt": "it"
}
```

### GPU Memory & Batch Size Recommendations

| GPU Model | VRAM | Recommended Batch Size | Notes |
|-----------|------|----------------------|-------|
| **RTX 5070** | **12GB** | **16** | **Tested configuration** |

**Note**: RTX 5070 with 12GB VRAM successfully runs with batch size 16. Start with recommended size and adjust based on your specific setup and memory usage.

### Optimization
- **Optimizer**: Adam (lr=1e-4, eps=1e-9)
- **Loss Function**: Cross-entropy with label smoothing (0.1)
- **Initialization**: Xavier initialization for better convergence
- **Padding**: Masked during loss calculation

## Metrics & Logging

### Training Metrics (TensorBoard)
- **`train/loss`**: Cross-entropy training loss
- **`train/learning_rate`**: Current learning rate

### Validation Metrics (TensorBoard)
- **`validation/loss`**: Cross-entropy validation loss
- **`validation/perplexity`**: Model uncertainty (exp(loss))
- **`validation/bleu`**: BLEU score (translation quality)
- **`validation/rouge_l`**: ROUGE-L score (semantic similarity)
- **`validation/cer`**: Character Error Rate
- **`validation/wer`**: Word Error Rate
- **`validation/avg_source_length`**: Average source sentence length
- **`validation/avg_target_length`**: Average target sentence length
- **`validation/avg_prediction_length`**: Average prediction length
- **`validation/dataset_size`**: Number of validation samples

## Evaluation Framework

### Comprehensive Evaluation
The evaluation framework provides:
- **Statistical Significance**: Evaluates entire validation/test sets
- **Multiple Metrics**: BLEU, ROUGE-L, CER, WER, Perplexity
- **Prediction Export**: Saves predictions for analysis (configurable limit)
- **Comparative Analysis**: Performance across different datasets

### Validation During Training
- **Full Dataset**: Evaluates entire validation set after each epoch
- **Progress Tracking**: Shows validation progress with tqdm
- **Example Display**: Shows sample translations for qualitative assessment
- **Comprehensive Logging**: All metrics logged to TensorBoard

## Visualization Tools

### Generated Plots
1. **`training_dashboard.png`**: Complete training overview
   - Training loss progression
   - Validation metrics comparison
   - BLEU score evolution
   - Error rate trends

2. **`training_loss_detailed.png`**: Detailed loss analysis
   - Raw and smoothed training loss
   - Linear and logarithmic scales

3. **`validation_metrics_detailed.png`**: Validation progression
   - Individual metric evolution
   - Epoch-by-epoch analysis

4. **`evaluation_comparison.png`**: Final model comparison
   - Validation vs test performance
   - Metric comparison bars

5. **`training_report.md`**: Comprehensive summary
   - Training statistics
   - Final evaluation results
   - Generated plots index

## Model Checkpointing

### Checkpoint Format
```python
{
    "epoch": 4,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "global_step": 12543
}
```

### Resuming Training
1. Edit `config.py`: Set `"preload_weights": "04"` for epoch 4
2. Run `python train.py`
3. Training resumes from the specified checkpoint

## Performance Expectations

### Validation Schedule
- **Quick Validation**: Sample examples shown during training
- **Full Validation**: Complete evaluation after each epoch
- **Test Evaluation**: Run separately with `evaluate.py`


### Memory Issues
- **Reduce batch size** in `config.py` (try reducing by 2-4 at a time)
- **Monitor GPU memory usage** during training with `nvidia-smi`
- **Sequence length** try reducing the sequence length
```