# NexText — Production-Grade Abstractive Text Summarization System

An end-to-end NLP and MLOps system that performs **abstractive dialogue summarization** using a fine-tuned **Google PEGASUS** transformer model on the **SAMSum** dataset. Built with **PyTorch**, **Hugging Face Transformers**, **FastAPI**, and a modern web interface.

---

## 📑 Table of Contents

1. [Project Overview & Elevator Pitches](#1-project-overview--elevator-pitches)
   - [30-Second Summary](#30-second-pitch)
   - [1-Minute Summary](#1-minute-pitch)
   - [3-Minute In-Depth Walkthrough](#3-minute-in-depth-walkthrough)
2. [Problem Statement & Why Text Summarization](#2-problem-statement--why-text-summarization)
   - [Extractive vs. Abstractive Summarization](#extractive-vs-abstractive-summarization)
   - [Challenges in Dialogue Summarization](#challenges-in-dialogue-summarization)
3. [Core NLP & Transformer Concepts](#3-core-nlp--transformer-concepts)
   - [Sequence-to-Sequence (Seq2Seq) Architecture](#sequence-to-sequence-seq2seq-architecture)
   - [Self-Attention & Cross-Attention](#self-attention--cross-attention)
   - [Subword Tokenization (SentencePiece / Unigram)](#subword-tokenization-sentencepiece--unigram)
   - [Decoding Strategies: Greedy vs. Beam Search](#decoding-strategies-greedy-vs-beam-search)
   - [Evaluation Metrics: ROUGE-1, ROUGE-2, ROUGE-L](#evaluation-metrics-rouge-1-rouge-2-rouge-l)
4. [Model Architecture: Google PEGASUS](#4-model-architecture-google-pegasus)
   - [Why PEGASUS for Summarization?](#why-pegasus-for-summarization)
   - [Gap Sentences Generation (GSG) Pre-training Objective](#gap-sentences-generation-gsg-pre-training-objective)
   - [Base Checkpoint & Fine-Tuning Strategy](#base-checkpoint--fine-tuning-strategy)
5. [Dataset & Data Engineering: SAMSum](#5-dataset--data-engineering-samsum)
   - [Dataset Characteristics & Splits](#dataset-characteristics--splits)
   - [Data Transformation & Tokenization Pipeline](#data-transformation--tokenization-pipeline)
6. [End-to-End System Architecture & Data Flow](#6-end-to-end-system-architecture--data-flow)
7. [ML Pipeline Stages (Detailed Walkthrough)](#7-ml-pipeline-stages-detailed-walkthrough)
   - [Stage 1: Data Ingestion](#stage-1-data-ingestion)
   - [Stage 2: Data Validation](#stage-2-data-validation)
   - [Stage 3: Data Transformation](#stage-3-data-transformation)
   - [Stage 4: Model Training](#stage-4-model-training)
   - [Stage 5: Model Evaluation](#stage-5-model-evaluation)
8. [Configuration Management & Design Patterns](#8-configuration-management--design-patterns)
9. [Training Hyperparameters & GPU Optimization](#9-training-hyperparameters--gpu-optimization)
10. [Model Serving, FastAPI Architecture & Endpoints](#10-model-serving-fastapi-architecture--endpoints)
    - [FastAPI Lifespan Model Management](#fastapi-lifespan-model-management)
    - [Request Validation & Error Handling](#request-validation--error-handling)
    - [REST API Endpoints](#rest-api-endpoints)
11. [Frontend Interface & Client-Side Logic](#11-frontend-interface--client-side-logic)
12. [Project Directory Structure](#12-project-directory-structure)
13. [Key Code Snippets & Explanations](#13-key-code-snippets--explanations)
14. [Testing Suite & Validation Strategy](#14-testing-suite--validation-strategy)
15. [Setup, Installation & Execution Guide](#15-setup-installation--execution-guide)
16. [Logging, Error Handling & Observability](#16-logging-error-handling--observability)
17. [Common Bugs, Debugging Lessons & Solutions](#17-common-bugs-debugging-lessons--solutions)
18. [Performance Considerations, Security & Limitations](#18-performance-considerations-security--limitations)
19. [Deployment Architecture & Docker Strategy](#19-deployment-architecture--docker-strategy)
20. [Future Improvements](#20-future-improvements)
21. [Interview Quick-Revision Cheat Sheet](#21-interview-quick-revision-cheat-sheet)

---

## 1. Project Overview & Elevator Pitches

### 30-Second Pitch
> "NexText is an end-to-end NLP summarization platform that takes unstructured multi-party dialogues and generates concise, abstractive summaries using a fine-tuned Google PEGASUS transformer. It features a modular 5-stage MLOps pipeline—from ingestion and schema validation to tokenization, memory-optimized training with Adafactor and gradient accumulation, and automated ROUGE evaluation. The model is served via a high-performance FastAPI backend with lifespan caching and Pydantic validation, paired with a modern web UI."

### 1-Minute Pitch
> "In conversational text like chat logs, meetings, and customer support transcripts, information is noisy, colloquial, and scattered across multiple speakers. Extractive summarization fails here because individual sentences lack standalone context. 
> 
> To solve this, I built NexText. I fine-tuned `google/pegasus-cnn_dailymail` on the SAMSum dialogue corpus. PEGASUS is uniquely suited for abstractive summarization due to its Gap Sentences Generation (GSG) pre-training objective. Because transformer fine-tuning is memory-intensive, I implemented Adafactor optimization, gradient accumulation (16 steps), and gradient checkpointing to train efficiently within constrained VRAM.
> 
> The system is architected around clean software engineering principles: configuration-driven pipelines via YAML and typed dataclasses, singleton model loading in FastAPI lifespan context to prevent redundant GPU memory allocation, strict input validation, 9 automated test suites, structured logging, and an interactive frontend with real-time word reduction metrics."

### 3-Minute In-Depth Walkthrough
> "The NexText project addresses conversational summarization through an enterprise-structured machine learning lifecycle:
> 
> 1. **Problem & Motivation**: Natural dialogue has fragmented syntax, informal discourse markers, and conversational turn-taking that cannot be summarized simply by extracting sentences. Abstractive summarization synthesizes dialogue into third-person coherent statements.
> 
> 2. **Model Selection**: Standard masked language models (like BERT) predict random tokens. PEGASUS, however, is pre-trained with Gap Sentences Generation (GSG), where entire principal sentences are masked and reconstructed, making its internal representations pre-adapted for abstractive summarization.
> 
> 3. **Modular Pipeline Architecture**:
>    - **Data Ingestion**: Downloads and unzips raw data into an isolated artifact directory.
>    - **Data Validation**: Enforces deterministic pipeline execution by validating directory schemas (`train`, `test`, `validation`) and logging status files before downstream processing.
>    - **Data Transformation**: Tokenizes dialogues (max 512 tokens) and summaries (max 128 tokens) using the PEGASUS Unigram/SentencePiece tokenizer and serializes the processed Arrow dataset to disk.
>    - **Model Training**: Leverages Hugging Face `Trainer` configured with `adafactor` optimizer (which consumes significantly less memory than AdamW by factorizing second-moment matrices), mixed precision (`fp16`), gradient checkpointing, and gradient accumulation.
>    - **Model Evaluation**: Generates summaries over the test set using 4-beam search with length penalty (0.8) and computes ROUGE-1, ROUGE-2, and ROUGE-L metrics exported to CSV.
> 
> 4. **Production Serving**:
>    - The inference engine uses a dedicated `ModelPrediction` class that places the model into evaluation mode on CUDA/CPU.
>    - In `app.py`, FastAPI manages the model lifecycle using an `asynccontextmanager` (`@lifespan`), loading weights into memory exactly once at startup.
>    - Robust Pydantic request models filter empty strings, whitespace-only queries, and oversized payloads (>20,000 chars), with custom exception handlers returning clean JSON error responses.
> 
> 5. **Observability & Testing**: Automated testing with FastAPI `TestClient` covers 9 distinct edge cases, while custom timestamped file and stream handlers provide centralized traceability."

---

## 2. Problem Statement & Why Text Summarization

Modern digital communication produces massive volumes of conversational data: customer support chats, team discussions, medical consultations, and meeting transcripts. 

Extracting key decisions and actionable points manually is time-consuming and expensive. Automated text summarization solves this by compressing long conversational streams into compact overviews.

```
+-------------------------------------------------------------------------------+
| Raw Input Dialogue (Multi-party, colloquial, scattered)                       |
| "Hannah: Hey, are you coming to the dinner tonight?                          |
|  Rob: Yes, what time?                                                         |
|  Hannah: Around 7:30 PM at Luigi's.                                           |
|  Rob: Sounds good, see you there!"                                            |
+-------------------------------------------------------------------------------+
                                      |
                                      v [NexText PEGASUS Engine]
+-------------------------------------------------------------------------------+
| Generated Abstractive Summary (3rd person, syntactically synthesized)         |
| "Rob is meeting Hannah for dinner at Luigi's restaurant at 7:30 PM tonight."  |
+-------------------------------------------------------------------------------+
```

### Extractive vs. Abstractive Summarization

| Feature | Extractive Summarization | Abstractive Summarization (NexText) |
| :--- | :--- | :--- |
| **Mechanism** | Identifies, ranks, and extracts existing sentences directly from source text. | Generates new words, paraphrases concepts, and constructs novel sentences. |
| **Grammar & Flow** | Often disjointed; lacks pronoun resolution and context continuity. | Natural, cohesive, and syntactically fluid. |
| **Suitability for Chat** | **Poor**: Chat turns like *"Yes, 7:30 PM"* mean nothing in isolation. | **High**: Synthesizes speaker intent: *"Rob confirmed 7:30 PM"*. |
| **Model Complexity** | Lightweight (TextRank, TF-IDF, sentence classifiers). | Complex Transformer Seq2Seq models (PEGASUS, T5, BART). |
| **Risk** | Zero factual hallucination, but low readability. | Potential for hallucination if not properly tuned and bounded. |

### Challenges in Dialogue Summarization
1. **Turn-taking Structure**: Multi-speaker exchanges require tracking speaker identities (`Speaker A: ...`, `Speaker B: ...`).
2. **Ellipsis & Anaphora**: Pronouns ("he", "that", "it") reference entities mentioned several turns prior.
3. **Informal Syntax**: Slang, emojis, abbreviations, typos, and fragmented sentences degrade traditional NLP parsers.
4. **Information Density**: Meaning is distributed over short bursts rather than concentrated in topic sentences.

---

## 3. Core NLP & Transformer Concepts

### Sequence-to-Sequence (Seq2Seq) Architecture
A Seq2Seq model maps an input sequence $X = (x_1, x_2, ..., x_n)$ to an output sequence $Y = (y_1, y_2, ..., y_m)$, where $n \neq m$. It consists of:
- **Encoder**: Converts input tokens into continuous contextual latent representations.
- **Decoder**: Autoregressively generates target tokens conditioned on both previous decoder outputs and the encoder representations.

```
Encoder Input:  [Token_1] -> [Token_2] -> [Token_3] ... -> [Token_N]
                    |            |            |                 |
                +---------------------------------------------------+
                |            Bi-Directional Self-Attention          |
                +---------------------------------------------------+
                                          |
                                 Contextual Embeddings
                                          |
Decoder Output:                           v
  [<BOS>] ----> [Decoder Layer (Masked Self-Attn + Cross-Attn)] ----> Token 1
  Token 1 ----> [Decoder Layer (Masked Self-Attn + Cross-Attn)] ----> Token 2
  Token 2 ----> [Decoder Layer (Masked Self-Attn + Cross-Attn)] ----> [<EOS>]
```

### Self-Attention & Cross-Attention
- **Self-Attention**: Computes dynamic similarity scores between every token pair within the same sequence:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  Allows the model to resolve coreferences (e.g., matching "Luigi's" to "dinner").
- **Cross-Attention**: The decoder queries ($Q_{dec}$) attend to the encoder keys ($K_{enc}$) and values ($V_{enc}$), transferring information from the source dialogue to the summary.

### Subword Tokenization (SentencePiece / Unigram)
PEGASUS uses SentencePiece tokenization based on a Unigram language model. SentencePiece treats whitespace as a normal character (represented as ` `), preventing language-specific pre-tokenizers from failing on messy text or contractions.

### Decoding Strategies: Greedy vs. Beam Search
When generating tokens at inference time:
- **Greedy Search**: Selects $\arg\max P(w_t | w_{<t})$ at every step. Fast, but vulnerable to sub-optimal local maxima.
- **Beam Search ($k=4$)**: Maintains the top $k$ most probable partial hypotheses at each generation step, exploring multiple sentence structures and selecting the global highest-probability sequence.
- **Length Penalty ($\alpha = 0.8$)**: Modifies the log-probability score by sequence length:
  $$\text{Score}(Y) = \frac{\log P(Y)}{(\text{Length}(Y))^\alpha}$$
  A penalty $< 1.0$ prevents the model from generating unnecessarily verbose summaries while avoiding excessively short truncations.

### Evaluation Metrics: ROUGE-1, ROUGE-2, ROUGE-L
**ROUGE** (*Recall-Oriented Understudy for Gisting Evaluation*) measures n-gram overlap between machine-generated summaries ($C$) and human reference summaries ($R$):

- **ROUGE-1**: Overlap of unigrams (individual words). Measures content coverage:
  $$\text{ROUGE-1 Recall} = \frac{\sum_{w \in R} \text{Count}_{\text{match}}(w)}{\sum_{w \in R} \text{Count}(w)}$$
- **ROUGE-2**: Overlap of bigrams (pairs of consecutive words). Measures fluency and syntactic phrasing.
- **ROUGE-L**: Longest Common Subsequence (LCS). Evaluates sentence-level structure without requiring consecutive matches.

---

## 4. Model Architecture: Google PEGASUS

**PEGASUS** (*Pre-training with Extracted Gap-sentences for Abstractive SUmmarization Sequence-to-sequence*) was introduced by Google Research (Zhang et al., 2020) specifically engineered for abstractive text summarization.

```
Raw Document:
[ Sentence 1: Important thesis statement. ]  <-- Masked out (Gap Sentence)
[ Sentence 2: Supporting background detail. ]
[ Sentence 3: Key conclusion statement.   ]  <-- Masked out (Gap Sentence)

Pre-training Target:
Reconstruct [Sentence 1] + [Sentence 3] using only [Sentence 2]
```

### Why PEGASUS for Summarization?
Most general language models (BART, T5) use Masked Language Modeling (MLM) or token-span masking. PEGASUS introduces **Gap Sentences Generation (GSG)**:
1. Entire important sentences are selected from documents based on high ROUGE-1 scores against the rest of the text.
2. These sentences are masked with `[MASK]` tokens.
3. The model is forced to reconstruct these missing sentences in the decoder.

This pre-training task mirrors the downstream task of abstractive summarization.

### Base Checkpoint & Fine-Tuning Strategy
- **Base Checkpoint**: `google/pegasus-cnn_dailymail`
- **Parameter Count**: ~568M parameters
- **Architecture**: 16 encoder layers, 16 decoder layers, 16 attention heads, hidden dimension 1024.
- **Transfer Learning**: The model already possesses strong summarization grammar from CNN/DailyMail (news articles); fine-tuning on SAMSum adapts the model's domain from formal news prose to multi-turn informal dialogue.

---

## 5. Dataset & Data Engineering: SAMSum

The **SAMSum** dataset (Samsung R&D Institute Poland) contains realistic, human-written chat conversations across diverse daily scenarios with third-person human-written summaries.

### Dataset Characteristics & Splits

| Split | Number of Dialogues | Purpose |
| :--- | :--- | :--- |
| **Train** | 14,732 | Parameter optimization |
| **Validation** | 818 | Hyperparameter tuning & validation loss check |
| **Test** | 818 | Unbiased final ROUGE evaluation |

### Data Format Example

```json
{
  "id": "13818545",
  "dialogue": "Hannah: Hey, are you coming to the dinner tonight?\nRob: Yes, what time?\nHannah: Around 7:30 PM at Luigi's.\nRob: Sounds good, see you there!",
  "summary": "Rob is joining Hannah for dinner at Luigi's at 7:30 PM."
}
```

### Data Transformation & Tokenization Pipeline
In the transformation stage, raw text strings are mapped to numeric tensors:

```
Dialogue String  ---> Tokenizer (max_length=512, truncation=True) ---> input_ids, attention_mask
Summary String   ---> Tokenizer (text_target, max_length=128, truncation=True) ---> labels
```

- `input_ids`: Integer token indices representing words and subwords.
- `attention_mask`: 1 for real tokens, 0 for padding tokens (prevents model from attending to blank space).
- `labels`: Target summary token indices used for cross-entropy loss computation in the decoder.

---

## 6. End-to-End System Architecture & Data Flow

```
+---------------------------------------------------------------------------------------+
|                                    ML TRAINING PIPELINE                               |
|                                                                                       |
|   [ Source URL ] ---> [ Stage 1: Data Ingestion ] ---> artifacts/data_ingestion/      |
|                                                               |                       |
|   [ status.txt ] <--- [ Stage 2: Data Validation ] <----------+                       |
|                                                               |                       |
|   [ HuggingFace ] --> [ Stage 3: Data Transformation ] <------+                       |
|   (Pegasus Tokenizer)                                         |                       |
|                                                               v                       |
|                                                artifacts/data_transformation/         |
|                                                               |                       |
|   [ params.yaml ] --> [ Stage 4: Model Trainer ] <------------+                       |
|   (Adafactor, fp16)                                           |                       |
|                                                               v                       |
|                                                artifacts/model_trainer/               |
|                                                ├── pegasus-samsum-model/              |
|                                                └── tokenizer/                         |
|                                                               |                       |
|   [ metrics.csv ] <-- [ Stage 5: Model Evaluation ] <---------+                       |
|   (ROUGE-1/2/L)                                                                       |
+---------------------------------------------------------------------------------------+

+---------------------------------------------------------------------------------------+
|                                PRODUCTION INFERENCE PIPELINE                          |
|                                                                                       |
|   Browser UI (HTML/CSS/JS)                                                            |
|          |                                                                            |
|          |  HTTP POST /predict  {"text": "Hannah: Hey..."}                            |
|          v                                                                            |
|   FastAPI Application (app.py)                                                        |
|          |                                                                            |
|          |  Pydantic Validation (1 <= len <= 20,000, non-whitespace)                  |
|          v                                                                            |
|   ModelPrediction Component (Singleton loaded at Lifespan)                            |
|          |                                                                            |
|          |  1. Tokenize Input (max_length=512, PyTorch tensors)                       |
|          |  2. Transfer to Device (CUDA / CPU)                                        |
|          |  3. model.generate(num_beams=4, max_new_tokens=128, length_penalty=0.8)   |
|          |  4. tokenizer.decode(skip_special_tokens=True)                             |
|          v                                                                            |
|   JSON Response  {"summary": "Rob is joining Hannah..."}                              |
|          |                                                                            |
|          v                                                                            |
|   Browser UI updates DOM + displays compression ratio metrics                         |
+---------------------------------------------------------------------------------------+
```

---

## 7. ML Pipeline Stages (Detailed Walkthrough)

The training workflow is structured as 5 independent, executable stages orchestrated via `main.py`.

### Stage 1: Data Ingestion
- **Component**: `DataIngestion` (`src/NexText/components/data_ingestion.py`)
- **Pipeline**: `DataIngestionTrainingPipeline` (`src/NexText/pipeline/stage1_data_ingestion.py`)
- **Action**: Downloads dataset archive from `source_URL` to `artifacts/data_ingestion/data.zip` and unzips it. If the local file already exists, download is skipped to ensure idempotency.

### Stage 2: Data Validation
- **Component**: `DataValidation` (`src/NexText/components/data_validation.py`)
- **Pipeline**: `DataValidationTrainingPipeline` (`src/NexText/pipeline/stage2_data_validation.py`)
- **Action**: Verifies that all required splits (`train`, `test`, `validation`) exist inside `artifacts/data_ingestion/samsum_dataset`. Writes validation result (`True`/`False`) into `artifacts/data_validation/status.txt`. If validation fails, the pipeline halts before computing transformations.

### Stage 3: Data Transformation
- **Component**: `DataTransformation` (`src/NexText/components/data_transformation.py`)
- **Pipeline**: `DataTransformationTrainingPipeline` (`src/NexText/pipeline/stage3_data_transformation.py`)
- **Action**: Initializes `AutoTokenizer` from `google/pegasus-cnn_dailymail`. Runs `.map()` in batched mode over the dataset, encoding dialogues into `input_ids` and `attention_mask` (512 tokens), and summaries into `labels` (128 tokens). Saves pre-tokenized Arrow dataset to `artifacts/data_transformation/samsum_dataset`.

### Stage 4: Model Training
- **Component**: `ModelTrainer` (`src/NexText/components/model_trainer.py`)
- **Pipeline**: `ModelTrainerTrainingPipeline` (`src/NexText/pipeline/stage4_model_trainer.py`)
- **Action**: Loads pre-trained `AutoModelForSeq2SeqLM`, initializes `DataCollatorForSeq2Seq`, configures `TrainingArguments` (Adafactor optimizer, gradient checkpointing, `fp16`), and executes `trainer.train()`. Persists fine-tuned weights and tokenizer to `artifacts/model_trainer/`.

### Stage 5: Model Evaluation
- **Component**: `ModelEvaluation` (`src/NexText/components/model_evaluation.py`)
- **Pipeline**: `ModelEvaluationTrainingPipeline` (`src/NexText/pipeline/stage5_model_evaluation.py`)
- **Action**: Loads fine-tuned model and test dataset. Generates predictions using batch chunks and beam search. Computes ROUGE-1, ROUGE-2, and ROUGE-L using `rouge_scorer.RougeScorer` and exports metrics to `artifacts/model_evaluation/metrics.csv`.

---

## 8. Configuration Management & Design Patterns

NexText follows strict decoupling between configuration and business logic using:
1. **`config/config.yaml`**: Filepaths, URLs, directory structures, and artifact endpoints.
2. **`params.yaml`**: Model training hyperparameters.
3. **`src/NexText/entity/config_entity.py`**: Immutable dataclasses (`@dataclass(frozen=True)`) ensuring type safety.
4. **`src/NexText/config/configuration.py`**: `ConfigurationManager` reads YAMLs using `ConfigBox` and supplies typed configuration entities to components.

```
config.yaml + params.yaml
           |
           v
ConfigurationManager (Parses YAML via ConfigBox, creates required dirs)
           |
           v
Typed Frozen Dataclass (DataIngestionConfig, ModelTrainerConfig, etc.)
           |
           v
Component (DataIngestion, ModelTrainer, etc.)
```

---

## 9. Training Hyperparameters & GPU Optimization

From `params.yaml` and `model_trainer.py`:

| Parameter | Value | Technical Rationale |
| :--- | :--- | :--- |
| `num_train_epochs` | `1` | Sufficient for transfer learning fine-tuning on domain-specific dialogue without catastrophic forgetting. |
| `per_device_train_batch_size` | `1` | Minimizes peak VRAM consumption for large Seq2Seq models. |
| `gradient_accumulation_steps`| `16` | Simulates an effective batch size of $1 \times 16 = 16$ without requiring physical VRAM for 16 simultaneous samples. |
| `warmup_steps` | `500` | Gradually increases learning rate to prevent early divergence. |
| `weight_decay` | `0.01` | L2 regularization applied to weights to prevent overfitting. |
| `optim` | `"adafactor"` | Factorizes the second-moment matrix in adaptive optimizers; reduces optimizer memory overhead from $\mathcal{O}(2N)$ to $\mathcal{O}(2\sqrt{N})$. |
| `gradient_checkpointing` | `True` | Trades compute for memory by recomputing activations during backward pass instead of storing all intermediate activations. |
| `fp16` | `torch.cuda.is_available()` | Uses 16-bit half-precision floating point tensors on GPU, doubling throughput and halving memory footprint. |
| `eval_steps` | `500` | Periodic validation evaluation frequency. |

### Evaluation Results on SAMSum Test Set

From `artifacts/model_evaluation/metrics.csv`:

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **ROUGE-1** | **0.4527** (45.27%) | Strong unigram keyword overlap between generated and reference summaries. |
| **ROUGE-2** | **0.2179** (21.79%) | Solid bigram coherence and multi-word phrase alignment. |
| **ROUGE-L** | **0.3578** (35.78%) | High structural sequence similarity and sentence flow. |

---

## 10. Model Serving, FastAPI Architecture & Endpoints

The web application is implemented in `app.py` using **FastAPI** and served with **Uvicorn**.

### FastAPI Lifespan Model Management
Loading a 568M-parameter transformer model on every incoming HTTP request causes massive latency and GPU out-of-memory errors. NexText uses FastAPI's `lifespan` context manager:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    logger.info("Initializing ModelPrediction component on application startup...")
    predictor = ModelPrediction()  # Loaded ONCE into memory/VRAM
    yield
    logger.info("Shutting down NexText Application...")
```

### Request Validation & Error Handling
- **Pydantic Validation**: `PredictionRequest` guarantees:
  - Input is a string.
  - Text is not empty or composed solely of whitespace.
  - Text does not exceed 20,000 characters.
- **Custom Exception Handlers**:
  - `RequestValidationError` $\rightarrow$ Returns HTTP `400 Bad Request` with sanitized error messages.
  - `HTTPException` $\rightarrow$ Handles designated operational errors.
  - `Exception` $\rightarrow$ Intercepts unexpected server errors and returns clean `500 Internal Server Error` without exposing stack traces.

### REST API Endpoints

| Method | Endpoint | Description | Request Body | Response Body |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | Serves interactive web UI | None | HTML page |
| `GET` | `/health` | Health and GPU device status | None | `{"status": "healthy", "device": "cuda", "model": "Pegasus-SamSum"}` |
| `GET` | `/docs` | OpenAPI / Swagger interactive docs | None | HTML/JS Swagger UI |
| `POST` | `/predict` | Generates summary | `{"text": "..."}` | `{"summary": "..."}` |

---

## 11. Frontend Interface & Client-Side Logic

The user interface is built with semantic HTML5 (`templates/index.html`), modern CSS with glassmorphism and gradient accents (`static/style.css`), and vanilla JavaScript (`static/script.js`).

### Key Frontend Capabilities
1. **Interactive Counters**: Real-time character and word counters update on every keystroke.
2. **Preset Sample Dialogues**: Instant one-click loading of realistic conversation samples (Dialogue, Product Review, Project Meeting).
3. **Keyboard Shortcuts**: <kbd>Ctrl</kbd> + <kbd>Enter</kbd> (or <kbd>Cmd</kbd> + <kbd>Enter</kbd>) triggers instant summarization.
4. **Compression Ratio Analytics**: Computes and displays original word count, summary word count, and text reduction percentage (e.g., `72% reduction`).
5. **One-Click Clipboard Export**: Copies generated summaries to the system clipboard with toast notifications.
6. **Dynamic State Management**: Seamless transitions between Empty State, Loading Spinner, and Summary Content.

---

## 12. Project Directory Structure

```
NexText/
├── .github/
│   └── workflows/
│       └── .gitkeep                 # CI/CD workflow directory
├── artifacts/                       # Generated pipeline artifacts (Git-ignored in production)
│   ├── data_ingestion/
│   │   ├── data.zip                 # Downloaded dataset archive
│   │   └── samsum_dataset/          # Unzipped dataset splits (train, test, val)
│   ├── data_transformation/
│   │   └── samsum_dataset/          # Tokenized Arrow dataset
│   ├── data_validation/
│   │   └── status.txt               # Validation status log (True/False)
│   ├── model_evaluation/
│   │   └── metrics.csv              # Calculated ROUGE scores
│   └── model_trainer/
│       ├── pegasus-samsum-model/    # Fine-tuned model weights (config.json, pytorch_model.bin)
│       └── tokenizer/               # Tokenizer files (spiece.model, tokenizer_config.json)
├── config/
│   └── config.yaml                  # Pipeline filepaths and directory configuration
├── logs/
│   └── YYYY_MM_DD_HH_MM_SS.log      # Timestamped application and pipeline log files
├── research/                        # Jupyter experimentation notebooks
│   ├── 01_data_ingestion.ipynb
│   ├── 02_data_validation.ipynb
│   ├── 03_data_transformtion.ipynb
│   ├── 04_model__trainer.ipynb
│   └── 05_odel___evaluation.ipynb
├── src/
│   └── NexText/                     # Core NexText Python package
│       ├── __init__.py
│       ├── components/              # Modular ML execution components
│       │   ├── __init__.py
│       │   ├── data_ingestion.py
│       │   ├── data_validation.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   ├── model_evaluation.py
│       │   └── model_prediction.py  # Inference execution component
│       ├── config/                  # Configuration reader & validator
│       │   ├── __init__.py
│       │   └── configuration.py
│       ├── constants/               # Global static path constants
│       │   └── __init__.py
│       ├── entity/                  # Dataclass entity schemas
│       │   ├── __init__.py
│       │   └── config_entity.py
│       ├── logging/                 # Centralized logging setup
│       │   └── __init__.py
│       ├── pipeline/                # Stage orchestrators
│       │   ├── __init__.py
│       │   ├── stage1_data_ingestion.py
│       │   ├── stage2_data_validation.py
│       │   ├── stage3_data_transformation.py
│       │   ├── stage4_model_trainer.py
│       │   ├── stage5_model_evaluation.py
│       │   └── prediction.py
│       └── utils/                   # Shared utility helpers
│           ├── __init__.py
│           └── common.py            # YAML parser, dir creator, size calculator
├── static/                          # Frontend assets
│   ├── script.js                    # UI interaction & async API fetch handling
│   └── style.css                    # Modern UI styling & animations
├── templates/
│   └── index.html                   # Jinja2 HTML web interface
├── app.py                           # FastAPI application & server entry point
├── Dockerfile                       # Container deployment definition
├── main.py                          # Full 5-stage training pipeline runner
├── params.yaml                      # Training hyperparameters configuration
├── requirements.txt                 # Project dependencies
├── setup.py                         # Package installation setup
├── test_app.py                      # 9 automated API unit/integration tests
└── test_prediction.py               # Standalone inference verification script
```

---

## 13. Key Code Snippets & Explanations

### Snippet 1: Inference Engine (`src/NexText/components/model_prediction.py`)

```python
class ModelPrediction:
    def __init__(self):
        self.model_path = "artifacts/model_trainer/pegasus-samsum-model"
        self.tokenizer_path = "artifacts/model_trainer/tokenizer"

        # 1. Dynamically select GPU if CUDA is available, otherwise fallback to CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 2. Load tokenizer and fine-tuned model from serialized artifact directories
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)

        # 3. Clear conflicting legacy generation limits to honor max_new_tokens
        self.model.generation_config.max_length = None

        # 4. Transfer model parameters to target device and set evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()  # Disables dropout layers for deterministic inference

    def predict(self, text):
        # 5. Tokenize input text with truncation and PyTorch tensor return
        inputs = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        # 6. Transfer token tensors to device (GPU/CPU)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # 7. Disable gradient calculation to save memory and accelerate generation
        with torch.no_grad():
            summary_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,    # Cap output summary length
                num_beams=4,           # 4-beam search for higher quality phrasing
                length_penalty=0.8,    # Soft penalty favoring concise summaries
                early_stopping=True    # Halt generation when EOS token is reached
            )

        # 8. Decode integer token IDs into clean human-readable text
        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return summary
```

### Snippet 2: Configuration Entity Pattern (`src/NexText/config/configuration.py`)

```python
class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH
    ):
        # 1. Parse YAML files into attribute-accessible ConfigBox objects
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        # 2. Ensure root artifacts directory exists
        create_directories([self.config.artifacts_root])
```

*Why this matters*: Eliminates hardcoded strings across files. Changing a dataset URL, batch size, or output directory only requires modifying `config.yaml` or `params.yaml`.

---

## 14. Testing Suite & Validation Strategy

The automated test suite in `test_app.py` uses `fastapi.testclient.TestClient` to validate the application end-to-end without launching an external HTTP server.

```bash
python test_app.py
```

### Test Case Coverage Matrix

| # | Test Case | Target Tested | Expected Status | Result |
| :---: | :--- | :--- | :---: | :---: |
| **1** | Frontend UI Rendering | `GET /` | `200 OK` | **PASS** |
| **2** | Service Health Check | `GET /health` | `200 OK` | **PASS** |
| **3** | Valid Dialogue Input | `POST /predict` with standard dialogue | `200 OK` | **PASS** |
| **4** | Empty String Input | `POST /predict` with `""` | `400 Bad Request` | **PASS** |
| **5** | Whitespace-Only Input | `POST /predict` with `"   \n\t  "` | `400 Bad Request` | **PASS** |
| **6** | Missing Payload Key | `POST /predict` with `{"invalid_key": "..."}` | `400/422` | **PASS** |
| **7** | Invalid Data Type | `POST /predict` with `{"text": 12345}` | `400/422` | **PASS** |
| **8** | Long Valid Input | `POST /predict` with large conversational block | `200 OK` | **PASS** |
| **9** | Excessive Input Length | `POST /predict` with $>20,000$ characters | `400/422` | **PASS** |

---

## 15. Setup, Installation & Execution Guide

### Prerequisites
- Python 3.11+
- CUDA 11.8+ / 12.x compatible GPU (Optional; CPU fallback supported)

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/sohamnemade21/NexText.git
cd NexText

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies & Package
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Step 3: Run the FastAPI Web Application
```bash
python app.py
```
Open **http://127.0.0.1:8080** in your browser to interact with the UI, or visit **http://127.0.0.1:8080/docs** for interactive Swagger API documentation.

### Step 4: Run Tests
```bash
# Run automated API test suite
python test_app.py

# Run standalone inference verification
python test_prediction.py
```

### Step 5: (Optional) Re-run ML Training Pipeline
> **Note**: Pre-trained weights and tokenizer artifacts are already present in `artifacts/model_trainer/`. To re-execute training from scratch:
```bash
python main.py
```

---

## 16. Logging, Error Handling & Observability

### Logging Implementation (`src/NexText/logging/__init__.py`)
- Every pipeline execution and API request writes to dual destinations:
  1. **Console (`sys.stdout`)**: Real-time terminal feedback.
  2. **Timestamped File (`logs/YYYY_MM_DD_HH_MM_SS.log`)**: Persistent log storage.
- Standard format:
  ```
  [ 2026-09-25 10:30:15,123 ] 165 app - INFO - Received summarization request with 142 characters.
  ```

### Defensive Error Handling
- **Missing Dataset Directory**: `DataValidation` checks directory tree before transformation, preventing corrupted dataset writes.
- **Pipeline Abort on Validation Failure**: `main.py` explicitly raises an exception if `validation_status == False`.
- **Global API Handlers**: Any unhandled server exception is trapped by `general_exception_handler` in `app.py`, logged via `logger.exception()`, and returned to the client as a clean HTTP 500 JSON message.

---

## 17. Common Bugs, Debugging Lessons & Solutions

### 1. GPU / CUDA Out-of-Memory (OOM) During Training
- **Issue**: Standard fine-tuning of PEGASUS (~568M params) with AdamW optimizer required >16 GB VRAM, causing CUDA OOM on consumer GPUs.
- **Root Cause**: AdamW maintains two states per parameter (first and second moments $\approx 8$ bytes/param), consuming massive memory.
- **Solution**:
  1. Switched to `adafactor` optimizer in `TrainingArguments`, drastically reducing optimizer state footprint.
  2. Enabled `gradient_checkpointing=True` to trade recomputation time for activation memory.
  3. Set `per_device_train_batch_size=1` with `gradient_accumulation_steps=16`.
  4. Enabled `fp16=True` for mixed-precision computation.

### 2. Conflicting `max_length` in Hugging Face Generation
- **Issue**: Warning/error during `model.generate()`: `Both max_new_tokens and max_length are set`.
- **Root Cause**: Pre-trained PEGASUS configs have a default `max_length=142` embedded in `generation_config`.
- **Solution**: Added `self.model.generation_config.max_length = None` in `ModelPrediction.__init__()` before calling `generate(max_new_tokens=128)`.

### 3. Absolute vs. Relative Path Breakage Across Modules
- **Issue**: Executing scripts from subdirectories broke relative paths like `config/config.yaml`.
- **Solution**: Established a deterministic `PROJECT_ROOT = Path(__file__).resolve().parents[3]` in `src/NexText/constants/__init__.py` and anchored all config and artifact paths to `PROJECT_ROOT`.

### 4. Pydantic Error Prefixing in API Responses
- **Issue**: Pydantic v2 automatically prepends `"Value error, "` to custom validation messages.
- **Solution**: Added string sanitization inside `validation_exception_handler` in `app.py` to strip technical prefixes and return human-friendly error details.

---

## 18. Performance Considerations, Security & Limitations

### Performance Optimizations
1. **Model Singleton**: Loaded once during FastAPI lifespan, eliminating multi-second per-request initialization lag.
2. **Inference with `torch.no_grad()`**: Deactivates autograd computation graph construction, saving ~30% RAM/VRAM during generation.
3. **Selective Token Truncation**: Capped input dialogues at 512 tokens to prevent quadratic $\mathcal{O}(N^2)$ self-attention memory explosions.

### Security Considerations
1. **Input Payload Bounding**: Strict 20,000-character ceiling prevents Denial of Service (DoS) via algorithmic complexity exhaustion.
2. **Input Sanitization**: Rejection of whitespace-only strings prevents unnecessary GPU compute cycles.
3. **CORS Policy**: Configured in `app.py` (`allow_origins=["*"]`) for development flexibility; can be restricted to designated production domain origins.

### Limitations
1. **Context Window Constraint**: Texts longer than 512 tokens (~400 words) are truncated, potentially dropping later conversational turns unless chunked.
2. **Abstractive Hallucination Risk**: As with all generative models, there is a minor probability of synthesizing unstated details if the dialogue is highly ambiguous.
3. **Single Language Support**: Fine-tuned exclusively on English dialogue datasets.

---

## 19. Deployment Architecture & Docker Strategy

### Containerization Strategy (Production Blueprint)
To containerize NexText for deployment to AWS ECS, Azure Container Apps, or Google Cloud Run:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Production Deployment Pipeline
```
Code Push (Git) ---> GitHub Actions (Lint + test_app.py) ---> Docker Build ---> Container Registry (ECR/GCR) ---> Cloud Hosting (AWS/GCP/Azure)
```

---

## 20. Future Improvements

1. **Quantization & Acceleration**: Implement 8-bit / 4-bit quantization (via `bitsandbytes` or ONNX Runtime / TensorRT-LLM) to achieve sub-100ms inference on CPU and edge devices.
2. **Hierarchical / Chunked Summarization**: Introduce Map-Reduce or iterative chunking pipelines to summarize 50+ page documents without 512-token truncation.
3. **Multi-Model Support**: Support dynamic model selection (e.g., PEGASUS, BART-large-CNN, T5, or Llama-3-Instruct).
4. **User Feedback Loop**: Add a rating button (👍/👎) in the web UI to collect human-in-the-loop (RLHF) correction data for continuous fine-tuning.
5. **Full CI/CD Automation**: Implement GitHub Actions pipeline executing automated unit tests and Docker image publishing on every main branch merge.

---

## 21. Interview Quick-Revision Cheat Sheet

```
+------------------------------------------------------------------------------------------------+
|                                  NEXTTEXT INTERVIEW CHEAT SHEET                                |
+------------------------------------------------------------------------------------------------+
| Model Used         | google/pegasus-cnn_dailymail (~568M params)                              |
| Pre-training Task  | Gap Sentences Generation (GSG) — masking & reconstructing full sentences   |
| Fine-Tuning Target | SAMSum Dataset (14,732 train / 818 val / 818 test dialogues)               |
| Evaluation Scores  | ROUGE-1: 0.4527 | ROUGE-2: 0.2179 | ROUGE-L: 0.3578                         |
| Memory Tricks      | Adafactor optimizer, Gradient Accumulation (16), Gradient Checkpointing   |
| Generation Setup   | Beam Search (num_beams=4), length_penalty=0.8, max_new_tokens=128         |
| Serving Stack      | FastAPI with Lifespan singleton model caching, Uvicorn, Jinja2, Vanilla JS |
| Validation Rules   | Pydantic v2 validation (non-empty, non-whitespace, max 20,000 characters)  |
| Design Pattern     | ConfigurationManager + Frozen Dataclasses + Component-Pipeline separation  |
| Automated Tests    | 9 test cases in test_app.py using FastAPI TestClient                      |
+------------------------------------------------------------------------------------------------+
```

### Core Talking Points to Master:
1. **Why PEGASUS instead of BERT?** BERT is an encoder-only model designed for classification/token labeling (MLM). PEGASUS is a Seq2Seq encoder-decoder designed specifically for text generation with its Gap Sentences Generation objective.
2. **Why not Extractive Summarization?** In multi-speaker dialogue, individual dialogue turns (e.g. *"Sounds good, see you there!"*) lack standalone meaning. Abstractive summarization creates coherent third-person summaries.
3. **How was memory managed during fine-tuning?** Used Adafactor optimizer (reduces 2nd-moment memory), gradient checkpointing (trades compute for activation RAM), batch size of 1 with gradient accumulation of 16 steps, and fp16 mixed precision.
4. **How is low inference latency maintained in FastAPI?** The model is loaded exactly once into VRAM/RAM on application startup using an `asynccontextmanager` (`@lifespan`), and inference runs inside `torch.no_grad()`.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
