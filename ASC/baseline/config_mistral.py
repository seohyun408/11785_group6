"""
Configuration for Mistral-7B experiment
"""
import os
from typing import Dict, Any

# API 
API_CONFIG = {
    'huggingface_token': os.getenv('HUGGINGFACE_TOKEN'),
    'wandb_token': os.getenv('WANDB_TOKEN'),
}

# Model - Mistral-7B
MODEL_CONFIG = {
    'model_name': "mistralai/Mistral-7B-Instruct-v0.2",
    'new_model': "/ocean/projects/cis250219p/slee37/mistral-7B-instruct-dpo",
    'my_dpo_model': "/ocean/projects/cis250219p/slee37/mistral-7B-instruct-dpo",
    'hub_model_name': "mistral-7B-instruct-dpo",
    'enable_hub_upload': False,  
}

# Training
TRAINING_CONFIG = {
    'per_device_train_batch_size': 16,
    'gradient_accumulation_steps': 2,
    'learning_rate': 5e-5,
    'max_steps': 300,
    'max_prompt_length': 512,
    'max_length': 768,
}

# LoRA 
LORA_CONFIG = {
    'r': 8,
    'alpha': 32,
    'dropout': 0.05,
}

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
