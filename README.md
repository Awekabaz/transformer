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
- **Batch Size**: 6 for training, 1 for validation/test

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
    "batch_size": 6,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "seq_len": 350,
    "d_model": 512,
    "train_size": 0.8,
    "val_size": 0.1,
    "language_src": "en",
    "language_tgt": "it"
}
```

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

### Evaluation Metrics (JSON Export)
```json
{
  "validation": {
    "bleu": 0.2543,
    "rouge_l": 0.4821,
    "cer": 0.3456,
    "wer": 0.5234,
    "perplexity": 12.34,
    "num_samples": 2041,
    "avg_src_length": 23.4,
    "avg_tgt_length": 25.1,
    "avg_pred_length": 24.8
  },
  "test": { ... }
}
```

## Evaluation Framework

### Comprehensive Evaluation
The evaluation framework provides:
- **Statistical Significance**: Evaluates entire validation/test sets
- **Multiple Metrics**: BLEU, ROUGE-L, CER, WER, Perplexity
- **Prediction Export**: Saves all predictions for analysis
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

### Training Progress
- **Initial Loss**: ~8-10
- **Converged Loss**: ~2-4
- **BLEU Score**: 0.15-0.35 (depends on dataset size and training time)
- **Training Time**: ~2-4 hours per epoch on GPU

### Validation Schedule
- **Quick Validation**: Sample examples shown during training
- **Full Validation**: Complete evaluation after each epoch
- **Test Evaluation**: Run separately with `evaluate.py`

## Common Issues & Solutions

### Empty Predictions
Three common bugs to check:
1. **Attention calculation**: Ensure `attention_probs @ value` not `attention_scores @ value`
2. **Causal mask**: Use `diagonal=0` not `diagonal=1` in `dataset.py:12`
3. **Model loading**: Add `model.load_state_dict(state["model_state_dict"])` after loading

### Memory Issues
- Reduce batch size in `config.py`
- Use gradient checkpointing for large models
- Monitor GPU memory usage during training

### Slow Training
- Increase batch size if memory allows
- Use mixed precision training
- Profile code to identify bottlenecks

## Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers datasets tokenizers
pip install torchmetrics tensorboard
pip install matplotlib seaborn pandas numpy
pip install tqdm pathlib
```