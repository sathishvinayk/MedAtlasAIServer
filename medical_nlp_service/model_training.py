"""
ClinicalBERT Fine-Tuning - STAGE 2: CLINICAL SPECIALIZATION
FIXED VERSION - Proper Resume from Stage 2
"""

import pandas as pd
import json
import os
from typing import List, Dict, Optional
import logging
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from datasets import Dataset, load_from_disk
import torch
import numpy as np
import time
from datetime import datetime
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ColabClinicalBERT:
    def __init__(self, base_model: str = "emilyalsentzer/Bio_ClinicalBERT"):
        self.base_model = base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 Using device: {self.device}")

    def log_gpu_memory(self, stage: str):
        """Log GPU memory usage for debugging"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"💾 GPU Memory [{stage}]: Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

    def clean_text(self, text: str) -> str:
        if not text or not isinstance(text, str): return ""
        cleaned = re.sub(r"[^a-zA-Z0-9.,;:!?()\[\]\-\s]", " ", text)
        return re.sub(r"\s+", " ", cleaned).strip()

    def create_colab_dataset(self, text_file: str, output_dir: str, max_examples=20000, validation_split=0.05):
        """Create dataset with memory-efficient chunking"""
        logger.info("📊 Creating dataset...")
        self.log_gpu_memory("dataset_creation_start")

        if not os.path.exists(text_file):
            logger.error(f"❌ File not found: {text_file}")
            return [], []

        with open(text_file, 'r') as f:
            lines = f.readlines()

        logger.info(f"📄 Found {len(lines)} lines")

        def read_in_chunks(file_path, chunk_size=5000):
            with open(file_path, 'r') as f:
                chunk = []
                for i, line in enumerate(f):
                    if line.strip():
                        cleaned = self.clean_text(line.strip())[:800]
                        if cleaned: chunk.append({"text": cleaned})
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
                    if len(chunk) + (i // chunk_size * chunk_size) >= max_examples:
                        break
                if chunk: yield chunk

        tokenizer = AutoTokenizer.from_pretrained(self.base_model, padding_side="right", model_max_length=512)

        train_chunks, val_chunks = [], []
        total_examples = 0

        for chunk_idx, chunk in enumerate(read_in_chunks(text_file)):
            if total_examples >= max_examples:
                logger.info(f"✅ Reached max examples limit ({max_examples})")
                break

            # Ensure we have enough examples for validation split
            if len(chunk) < 10:  # Skip very small chunks
                continue

            split_idx = max(1, int(len(chunk) * (1 - validation_split)))  # Ensure at least 1 train example
            train_chunk, val_chunk = chunk[:split_idx], chunk[split_idx:]

            def tokenize_function(examples):
                return tokenizer(examples["text"], truncation=True, padding=False, max_length=256, return_special_tokens_mask=True)

            # Process training chunk
            if train_chunk:
                train_dataset = Dataset.from_list(train_chunk)
                tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"], batch_size=500)
                train_file = f"{output_dir}/train_chunk_{chunk_idx}"
                tokenized_train.save_to_disk(train_file)
                train_chunks.append(train_file)
                total_examples += len(train_chunk)

            # Process validation chunk (only if we have enough examples)
            if val_chunk and len(val_chunk) >= 5:  # Minimum 5 examples for validation
                val_dataset = Dataset.from_list(val_chunk)
                tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"], batch_size=500)
                val_file = f"{output_dir}/val_chunk_{chunk_idx}"
                tokenized_val.save_to_disk(val_file)
                val_chunks.append(val_file)

            logger.info(f"✅ Chunk {chunk_idx}: {len(train_chunk)} train, {len(val_chunk)} val")
            self.log_gpu_memory(f"chunk_{chunk_idx}")

        logger.info(f"🎉 Created {len(train_chunks)} train chunks ({total_examples} examples) and {len(val_chunks)} val chunks")
        self.log_gpu_memory("dataset_creation_end")
        return train_chunks, val_chunks

    def train_on_colab(self, train_chunks, val_chunks, output_dir: str, resume_from_checkpoint=None):
        """Training with memory-efficient dataset loading - FIXED RESUME LOGIC"""
        logger.info("🎯 Starting STAGE 2 Training - Clinical Specialization...")
        self.log_gpu_memory("training_start")

        if not train_chunks:
            raise ValueError("No training data provided")

        tokenizer = AutoTokenizer.from_pretrained(self.base_model, padding_side="right")

        # 🎯 FIXED: SMART MODEL LOADING - Resume from Stage 2 if available
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            logger.info(f"🔄 RESUMING from Stage 2: {resume_from_checkpoint}")
            model = AutoModelForMaskedLM.from_pretrained(
                resume_from_checkpoint,
                trust_remote_code=True,
                ignore_mismatched_sizes=True
            )
        else:
            # Start fresh from Stage 1
            stage1_model_path = f"{os.path.dirname(output_dir)}/checkpoints"
            logger.info(f"🆕 Starting from Stage 1: {stage1_model_path}")
            model = AutoModelForMaskedLM.from_pretrained(stage1_model_path)

        model.to(self.device)

        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 🎯 TRAINING ARGUMENTS - Overwrite on every save
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,  # ✅ Overwrite files in checkpoints_stage2
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=3e-5,
            warmup_steps=500,
            weight_decay=0.01,
            logging_steps=50,
            save_steps=100,  # ✅ Save every 100 steps
            eval_steps=100,
            eval_strategy="steps" if val_chunks else "no",
            save_strategy="steps",
            load_best_model_at_end=False,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            seed=42,
            report_to="none",
            max_grad_norm=1.0,
            logging_first_step=True,
            save_total_limit=1,  # ✅ Keep only latest model
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        class MemoryEfficientDataset(torch.utils.data.Dataset):
            """Lazy-loading dataset to avoid memory explosion"""
            def __init__(self, chunk_files, dataset_type="train"):
                self.chunk_files = chunk_files
                self.dataset_type = dataset_type
                self.current_chunk = None
                self.current_chunk_idx = -1

                # Calculate total length without loading all chunks
                self._length = 0
                self.chunk_sizes = []
                for chunk_file in chunk_files:
                    try:
                        chunk = load_from_disk(chunk_file)
                        chunk_size = len(chunk)
                        self.chunk_sizes.append(chunk_size)
                        self._length += int(chunk_size)
                    except Exception as e:
                        logger.error(f"❌ Error loading chunk {chunk_file}: {e}")
                        self.chunk_sizes.append(0)

                self.cumulative_sizes = np.cumsum([0] + self.chunk_sizes)
                logger.info(f"📚 {dataset_type.upper()} Dataset: {self._length} examples across {len(chunk_files)} chunks")

            def __len__(self):
                return self._length

            def __getitem__(self, idx):
                idx = int(idx)
                current_idx = 0
                for chunk_idx, chunk_file in enumerate(self.chunk_files):
                    if chunk_idx != self.current_chunk_idx:
                        self.current_chunk = load_from_disk(chunk_file)
                        self.current_chunk_idx = chunk_idx

                    chunk_size = len(self.current_chunk)
                    if idx < current_idx + chunk_size:
                        local_idx = idx - current_idx
                        item = self.current_chunk[local_idx]
                        return {k: torch.tensor(v) for k, v in item.items()}

                    current_idx += chunk_size

                return {k: torch.tensor(v) for k, v in self.current_chunk[-1].items()}

        # Create datasets
        train_dataset = MemoryEfficientDataset(train_chunks, "train")

        if val_chunks:
            val_dataset = MemoryEfficientDataset(val_chunks, "validation")
            logger.info(f"✅ Using validation set with {len(val_dataset)} examples")
        else:
            val_dataset = None
            logger.warning("⚠️ No validation set available - training without validation")

        self.log_gpu_memory("after_dataset_creation")

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
        )

        logger.info("🎬 STAGE 2 Training started...")
        if resume_from_checkpoint:
            logger.info("🔄 RESUME MODE: Continuing from Stage 2 progress")
        else:
            logger.info("🆕 FRESH START: Starting from Stage 1")
        logger.info("🎯 Focus: Clinical specificity and medical pattern recognition")
        self.log_gpu_memory("training_loop_start")

        start_time = time.time()
        try:
            # 🎯 FIXED: Proper resume handling
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            try:
                trainer.save_model()
                logger.info("💾 Model saved despite error")
            except:
                pass
            raise e

        training_time = time.time() - start_time
        logger.info(f"🎉 STAGE 2 Training completed in {training_time:.0f}s ({training_time/60:.1f} minutes)")

        self.log_gpu_memory("training_end")

        # Save final model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        logger.info(f"💾 Stage 2 model saved to: {output_dir}")
        return output_dir

class ColabOrchestrator:
    def __init__(self, base_dir: str = "/content/drive/MyDrive/clinicalbert_training"):
        self.base_dir = base_dir
        self.fine_tuner = ColabClinicalBERT()

        # Create directory structure
        os.makedirs(f"{base_dir}/kaggle_data", exist_ok=True)
        os.makedirs(f"{base_dir}/dataset_chunks", exist_ok=True)
        os.makedirs(f"{base_dir}/checkpoints_stage2", exist_ok=True)

        logger.info(f"📁 Working directory: {base_dir}")

    def get_resume_checkpoint(self):
        """Check if we should resume from Stage 2"""
        stage2_dir = f"{self.base_dir}/checkpoints_stage2"

        # Check if Stage 2 has model files
        if os.path.exists(stage2_dir):
            files = os.listdir(stage2_dir)
            if any(f.endswith(('.bin', '.safetensors')) for f in files):
                logger.info("✅ Found Stage 2 progress - will RESUME from here")
                return stage2_dir  # Return the directory path to resume from

        logger.info("🆕 No Stage 2 progress found - starting fresh from Stage 1")
        return None

    def _extract_text(self, row):
        """Extract text from row - FIXED for 'Conversation' column"""
        if 'Conversation' in row and pd.notna(row['Conversation']):
            text = str(row['Conversation'])
            cleaned = self.fine_tuner.clean_text(text)
            if cleaned and len(cleaned) > 10:
                return cleaned[:1000]
        return ""

    def run_stage2_pipeline(self, train_csv_path: str, test_csv_path: str, max_examples=15000):
        """Run STAGE 2 training pipeline - Clinical Specialization"""
        logger.info("🚀 Starting ClinicalBERT STAGE 2 - Clinical Specialization")
        self.fine_tuner.log_gpu_memory("pipeline_start")

        try:
            # 🎯 CHECK STAGE 1 EXISTS
            stage1_path = f"{self.base_dir}/checkpoints"
            if not os.path.exists(stage1_path):
                raise ValueError("❌ Stage 1 model not found! Run Stage 1 first.")
            logger.info("✅ Stage 1 model found")

            # 🎯 GET RESUME CHECKPOINT (Stage 2 if exists)
            resume_checkpoint = self.get_resume_checkpoint()

            # Check if dataset chunks exist
            dataset_chunks_dir = f"{self.base_dir}/dataset_chunks"
            if os.path.exists(dataset_chunks_dir) and os.listdir(dataset_chunks_dir):
                logger.info("📚 Using existing dataset chunks")
                train_chunks = [f"{dataset_chunks_dir}/{f}" for f in os.listdir(dataset_chunks_dir) if f.startswith('train_chunk')]
                val_chunks = [f"{dataset_chunks_dir}/{f}" for f in os.listdir(dataset_chunks_dir) if f.startswith('val_chunk')]
                logger.info(f"📊 Found {len(train_chunks)} train chunks and {len(val_chunks)} val chunks")
            else:
                logger.info("🔨 Processing data from CSV files...")
                # ... [keep your existing data processing code] ...

            if not train_chunks:
                raise ValueError("❌ No training chunks available!")

            # 🎯 STAGE 2 TRAINING
            stage2_output_dir = f"{self.base_dir}/checkpoints_stage2"

            final_model = self.fine_tuner.train_on_colab(
                train_chunks,
                val_chunks,
                stage2_output_dir,
                resume_from_checkpoint=resume_checkpoint  # 🎯 PASS RESUME CHECKPOINT
            )

            logger.info(f"✅ STAGE 2 Training complete! Model saved to: {final_model}")
            return final_model

        except Exception as e:
            logger.error(f"❌ Stage 2 pipeline failed: {e}")
            raise e

# 🎯 STAGE 2 MAIN EXECUTION
if __name__ == "__main__":
    print("=" * 60)
    print("🏁 CLINICALBERT STAGE 2 - CLINICAL SPECIALIZATION")
    print("=" * 60)

    try:
        orchestrator = ColabOrchestrator()
        model_path = orchestrator.run_stage2_pipeline("train.csv", "test.csv", max_examples=15000)

        print(f"\n🎉 STAGE 2 SUCCESS! Your clinically specialized model is ready!")
        print(f"📁 Location: {model_path}")
        print("\n📋 Next steps:")
        print("1. Run clinical diagnostic tests")
        print("2. Test with DeepScribe-like scenarios")
        print("3. Evaluate medical terminology understanding")

    except Exception as e:
        print(f"❌ Stage 2 pipeline failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("- Make sure Stage 1 training completed successfully")
        print("- Check that train.csv and test.csv are available")
        print("- Verify you have enough GPU memory")
        import traceback
        traceback.print_exc()
