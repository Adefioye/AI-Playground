
import sys
import os
from transformers import AutoConfig, AutoModel, AutoTokenizer

# --- Step 1: Add Custom Model Code to Python Path ---
# We need to tell Python where to find the .py files that define your custom model.
# This adds the directory containing the model definitions to the system path.
# Note: This assumes you are running this script from the root of the 'random_experiment' directory.
project_root = os.path.abspath('.')
custom_model_path = os.path.join(project_root, 'shared-subspaces', 'subspace_decoder')
sys.path.insert(0, custom_model_path)

# --- Step 2: Import Custom Model Classes ---
# Now that the path is set, we can import your custom config and model classes.
try:
    from models.shared_space_config import SharedSpaceDecoderConfig
    from models.shared_space_decoder import SharedSpaceDecoderModel
except ImportError as e:
    print(f"Error importing custom model classes: {e}")
    print(f"Please ensure the path '{custom_model_path}' is correct and contains the model definition files.")
    sys.exit(1)

# --- Step 3: Register the Custom Architecture ---
# This officially "teaches" the transformers library about your custom model type.
print("Registering custom model architecture: 'shared_subspace_decoder'")
AutoConfig.register("shared_subspace_decoder", SharedSpaceDecoderConfig)
AutoModel.register(SharedSpaceDecoderConfig, SharedSpaceDecoderModel)

# --- Step 4: Load the Checkpoint ---
# Now that the architecture is registered, we can load the model as usual.
# The 'trust_remote_code=True' flag is still good practice for custom models.
checkpoint_path = os.path.join(project_root, 'SubspaceDecoder_mha')
print(f"Attempting to load checkpoint from: {checkpoint_path}")

if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint directory not found at '{checkpoint_path}'")
    print("Please ensure the 'SubspaceDecoder_mha' directory exists in your project root.")
    sys.exit(1)

try:
    model = AutoModel.from_pretrained(checkpoint_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    print("\nSuccessfully loaded model and tokenizer!")
    print("Model:", model.__class__.__name__)
except Exception as e:
    print(f"\nAn error occurred during model loading: {e}")

