"""
Configuration using dictionaries - another common Python pattern.
RTX 3090 기준 24G
"""

import os
from typing import Dict, Any

# =============================================================================
# Configuration Dictionaries
# =============================================================================

# API 
API_CONFIG = {
    'huggingface_token': os.getenv('HUGGINGFACE_TOKEN'),
    'wandb_token': os.getenv('WANDB_TOKEN'),
}

# Model
MODEL_CONFIG = {
    'model_name': "mistralai/Mistral-7B-Instruct-v0.2", # Original model
    'new_model': "/scratch/tjgus0408/mistral-7B-instruct-dpo", # New DPO model (Train!)
    'my_dpo_model': "/scratch/tjgus0408/mistral-7B-instruct-dpo", # Pre-trained DPO model
    'hub_model_name': "mistral-7B-instruct-dpo",  # Hub upload model
    'enable_hub_upload': False,  
}

# Training
TRAINING_CONFIG = {
    'per_device_train_batch_size': 4,
    'gradient_accumulation_steps': 8,
    'learning_rate': 5e-5,
    'max_steps': 300,
    'max_prompt_length': 768,
    'max_length': 1024,
}

# LoRA 
LORA_CONFIG = {
    'r': 16,
    'alpha': 16,
    'dropout': 0.05,
}

# =============================================================================
# Utility Functions
# =============================================================================

def validate_tokens() -> bool:
    """Validate that required API tokens are available."""
    missing_tokens = []
    
    for key, value in API_CONFIG.items():
        if not value:
            missing_tokens.append(key.upper())
    
    if missing_tokens:
        print(f"Warning: Missing environment variables: {', '.join(missing_tokens)}")
        return False
    return True

def get_all_config() -> Dict[str, Any]:
    """Get all configuration as a single dictionary."""
    return {
        'api': API_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'lora': LORA_CONFIG,
    }

def update_config(section: str, updates: Dict[str, Any]) -> None:
    """Update a specific configuration section."""
    if section == 'api':
        API_CONFIG.update(updates)
    elif section == 'model':
        MODEL_CONFIG.update(updates)
    elif section == 'training':
        TRAINING_CONFIG.update(updates)
    elif section == 'lora':
        LORA_CONFIG.update(updates)
    else:
        raise ValueError(f"Unknown configuration section: {section}")
