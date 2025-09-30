# How to Package a Custom `transformers` Model

This guide provides the complete steps to package a custom model, like `shared_subspace_decoder`, so that it can be loaded with `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` and shared on the Hugging Face Hub.

The `trust_remote_code` feature has three main requirements:
1.  All custom Python source files must be in the same root directory as the model weights.
2.  All imports between these custom files must be **explicitly relative** (e.g., `from .my_local_file import ...`).
3.  The `config.json` file must contain an `auto_map` that tells `transformers` where to find your custom configuration and model classes.

---

### Step 1: Create a Clean Checkpoint Directory

It's best to start fresh. Create a new directory to hold the final, correct package.

```bash
mkdir MyCustomModel
```

### Step 2: Copy Model Weights & Tokenizer

This step is manual. Find your trained model weights and tokenizer files and copy them into the new directory.

- **Model Weights**: `pytorch_model.bin`, `optimizer.pt`, etc.
- **Tokenizer Files**: `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, etc.

### Step 3: Copy All Necessary Python Source Files

The `SharedSpaceDecoderForCausalLM` model architecture is defined across several files in your `shared-subspaces/subspace_decoder/` directory. Run the following commands to copy all of them into the root of your new `MyCustomModel` directory.

```bash
# Copy the main model and config definitions
cp shared-subspaces/subspace_decoder/models/shared_space_config.py MyCustomModel/
cp shared-subspaces/subspace_decoder/models/shared_space_decoder.py MyCustomModel/

# Copy all of the layer definitions
cp shared-subspaces/subspace_decoder/layers/*.py MyCustomModel/
```

### Step 4: Fix All Python Imports

This is the most critical step. We must change the imports in the files you just copied to be explicitly relative. The `sed` command is a reliable way to do this.

**Run these commands exactly as written from your `random_experiment` directory:**

```bash
# Fix imports in task_heads.py
sed -i '' 's/from shared_space_config/from .shared_space_config/g' MyCustomModel/task_heads.py
sed -i '' 's/from shared_space_decoder/from .shared_space_decoder/g' MyCustomModel/task_heads.py

# Fix imports in shared_space_decoder.py
sed -i '' 's/from mla/from .mla/g' MyCustomModel/shared_space_decoder.py
sed -i '' 's/from feedforward/from .feedforward/g' MyCustomModel/shared_space_decoder.py
sed -i '' 's/from shared_space_config/from .shared_space_config/g' MyCustomModel/shared_space_decoder.py

# Fix imports in feedforward.py
sed -i '' 's/from models.shared_space_config/from .shared_space_config/g' MyCustomModel/feedforward.py

# Fix imports in mla.py
sed -i '' 's/from models.shared_space_config/from .shared_space_config/g' MyCustomModel/mla.py
```

### Step 5: Create the `config.json` File

This file tells `transformers` everything about your model. Create a new file named `config.json` inside the `MyCustomModel` directory with the following content. The `auto_map` is the most important part.

```json
{
  "architectures": [
    "SharedSpaceDecoderForCausalLM"
  ],
  "model_type": "shared_subspace_decoder",
  "auto_map": {
    "AutoConfig": "shared_space_config.SharedSpaceDecoderConfig",
    "AutoModelForCausalLM": "task_heads.SharedSpaceDecoderForCausalLM"
  },
  "attention_backend": "flash_attention_2",
  "attention_bias": false,
  "attention_dropout_prob": 0.1,
  "bos_token_id": 50256,
  "classifier_dropout": null,
  "dtype": "float32",
  "eos_token_id": 50256,
  "ffn_decompose": false,
  "ffn_rank": null,
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 2048,
  "kv_shared_dim": null,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 1024,
  "nope_dims": 32,
  "norm_type": "rmsnorm",
  "num_attention_heads": 12,
  "num_dense_layers": 0,
  "num_hidden_layers": 12,
  "o_shared_dim": null,
  "pad_token_id": 50256,
  "q_shared_dim": null,
  "qk_private_dim": 64,
  "rms_norm_eps": 1e-06,
  "rope_dims": 32,
  "rope_scaling": {
    "factor": 2.0,
    "type": "linear"
  },
  "rope_theta": 10000.0,
  "transformers_version": "4.56.0",
  "vo_private_dim": 64,
  "vocab_rank": null,
  "vocab_size": 50257,
  "vocab_subspace": false
}
```

### Step 6: Verify and Load

After following these steps, your `MyCustomModel` directory is now a self-contained, shareable checkpoint. You can test it locally.

**Your final directory structure should look like this:**
```
MyCustomModel/
├── config.json
├── feedforward.py
├── gla.py
├── mla.py
├── pytorch_model.bin
├── shared_space_config.py
├── shared_space_decoder.py
├── task_heads.py
└── tokenizer.json
(and other model/tokenizer files)
```

**You can now load it in Python:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make sure to use the correct Auto-class
model = AutoModelForCausalLM.from_pretrained("./MyCustomModel", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("./MyCustomModel")

print("Successfully loaded model!")
```

This directory can now be pushed to the Hugging Face Hub and shared with others.
