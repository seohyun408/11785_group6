import os
import gc
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from transformers import TrainerCallback

import bitsandbytes as bnb
import wandb
from typing import Dict, Optional
from datasets import Dataset, load_dataset
from tqdm import tqdm
import evaluate
import numpy as np
import torch.nn.functional as F
from torch.nn import KLDivLoss

from detoxify import Detoxify

import warnings
warnings.filterwarnings("ignore")

import torch.distributed as dist


# =============================================================================
# LLM
# =============================================================================

def main():
    LLM_model = os.getenv("LLM_MODEL").lower()

    checkpoint_dir = f"/ocean/projects/cis250219p/slee37/final_checkpoint/{LLM_model}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if LLM_model == 'llama':
        from config_llama import API_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, LORA_CONFIG, validate_tokens
        print("Using Llama3.1-8B configuration")
    elif LLM_model == 'qwen':
        from config_qwen import API_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, LORA_CONFIG, validate_tokens
        print("Using Qwen2.5-7B configuration")
    elif LLM_model == 'mistral':
        from config_mistral import API_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, LORA_CONFIG, validate_tokens
        print("Using Mistral-7B configuration")



    # =============================================================================
    # MULTI-GPU SETUP
    # =============================================================================
    
    def setup_ddp():
        """Initialize DDP if running in distributed mode."""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ.get('LOCAL_RANK', 0))

            # Set CUDA device BEFORE initializing the process group
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)

            # Ensure rendezvous address/port exist (torchrun usually provides these)
            os.environ.setdefault('MASTER_ADDR', os.environ.get('MASTER_ADDR', '127.0.0.1'))
            os.environ.setdefault('MASTER_PORT', os.environ.get('MASTER_PORT', '29500'))

            if not dist.is_initialized():
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=world_size,
                    rank=rank
                )
            
            # Verify initialization was successful
            if dist.is_initialized():
                print(f"DDP initialized successfully: rank={rank}, world_size={world_size}, local_rank={local_rank}")
            
            return True, rank, world_size, local_rank
        else:
            print('Not using distributed mode')
            return False, 0, 1, 0

    use_ddp, rank, world_size, local_rank = setup_ddp()
    # if LOCAL_RANK == 0:
    if use_ddp:
        print(f"Using DDP with {world_size} processes, rank {rank}, local_rank {local_rank}")
        if rank == 0:
            print(f"Available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


    # =============================================================================
    # SETUP / API
    # =============================================================================

    hf_token = API_CONFIG['huggingface_token']

    # Initialize wandb only on main process
    if not use_ddp or rank == 0:
        wandb.login(key=API_CONFIG['wandb_token'])
        
        if LLM_model.lower() == 'mistral':
            run_name = f"Baseline--mistral7B"
        elif LLM_model.lower() == 'qwen':
            run_name = f"Baseline--qwen2.5-7B"
        elif LLM_model.lower() == 'llama':
            run_name = f"Baseline--llama3.1-8B"
        
        wandb.init(
            project="IDL_11785_group6",
            name=run_name,
            config={
                "model_name": MODEL_CONFIG['model_name'],
                "new_model": MODEL_CONFIG['new_model'],
                "training_config": TRAINING_CONFIG,
                "lora_config": LORA_CONFIG,
                "num_epochs": 5,
                "learning_rate": TRAINING_CONFIG['learning_rate'],
                "batch_size": TRAINING_CONFIG['per_device_train_batch_size'],
            }
        )

    if LLM_model == 'mistral':
        run_name = f"Baseline--mistral7B"
    elif LLM_model == 'qwen':
        run_name = f"Baseline--qwen2.5-7B"
    elif LLM_model == 'llama':
        run_name = f"Baseline--llama3.1-8B"


    # =============================================================================
    # Functions
    # =============================================================================

    def extract_anthropic_prompt(prompt_and_response):
        """Extract the anthropic prompt from a prompt and response pair."""
        search_term = "\n\nAssistant:"
        search_term_idx = prompt_and_response.rfind(search_term)
        assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
        return prompt_and_response[: search_term_idx + len(search_term)]


    def get_hh(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: Optional[str] = None) -> Dataset:
        """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

        The dataset is converted to a dictionary with the following structure:
        {
            'prompt': List[str],
            'chosen': List[str],
            'rejected': List[str],
        }

        Prompts should be structured as follows:
        \n\nHuman: <prompt>\n\nAssistant:
        Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
        """
        dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
        if sanity_check: # For debugging
            dataset = dataset.select(range(min(len(dataset), 10000)))

        def split_prompt_and_responses(sample) -> Dict[str, str]:
            prompt = extract_anthropic_prompt(sample["chosen"])
            return {
                "prompt": prompt,
                "chosen": sample["chosen"][len(prompt) :],
                "rejected": sample["rejected"][len(prompt) :],
            }

        return dataset.map(split_prompt_and_responses)


    def calculate_kl_divergence(policy_model, reference_model, tokenizer, sample_texts, device="cuda"):
        """
        Calculate KL-divergence between policy and reference models on sample texts.
        (Average KL-divergence across all sample texts)
        """

        policy_model.eval()
        reference_model.eval()
        
        total_kl_div = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for text in sample_texts:
                
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                # For multi-GPU, let the model handle device placement
                if hasattr(policy_model, 'device'):
                    inputs = {k: v.to(policy_model.device) for k, v in inputs.items()}
                else:
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                
                policy_outputs = policy_model(**inputs)
                reference_outputs = reference_model(**inputs)
                
                policy_logits = policy_outputs.logits
                reference_logits = reference_outputs.logits
                
                policy_log_probs = F.log_softmax(policy_logits, dim=-1)
                reference_probs = F.softmax(reference_logits, dim=-1)
                
                reference_probs = reference_probs + 1e-8
                reference_probs = reference_probs / reference_probs.sum(dim=-1, keepdim=True)
                
                kl_div = F.kl_div(policy_log_probs, reference_probs, reduction='batchmean', log_target=False)
                
                if not torch.isnan(kl_div) and not torch.isinf(kl_div):
                    total_kl_div += kl_div.item()
                    num_samples += 1
                        
        
        return total_kl_div / num_samples if num_samples > 0 else 0.0


    # =============================================================================
    # DATASET
    # =============================================================================

    if use_ddp:
        if rank == 0:
            train_dataset = get_hh("train", sanity_check=False)
            eval_dataset = get_hh("test", sanity_check=False)
        if dist.is_initialized():
            # Use explicit device_ids to avoid NCCL choosing an unknown device
            try:
                dist.barrier(device_ids=[local_rank])
            except TypeError:
                # Older PyTorch: fall back to default barrier
                dist.barrier()
        if rank != 0:
            train_dataset = get_hh("train", sanity_check=False)
            eval_dataset = get_hh("test", sanity_check=False)
    else:
        train_dataset = get_hh("train", sanity_check=False)
        eval_dataset = get_hh("test", sanity_check=False)
    eval_dataset = eval_dataset.select(range(1000))

    train_dataset = train_dataset.with_format("torch")
    eval_dataset = eval_dataset.with_format("torch")


    # =============================================================================
    # Model
    # =============================================================================

    model_name = MODEL_CONFIG['model_name']
    new_model = MODEL_CONFIG["new_model"]
    my_dpo_model = MODEL_CONFIG["my_dpo_model"]

    push_new_model = MODEL_CONFIG["hub_model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Get token IDs from tokenizer to align with model config
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map=None if use_ddp else "auto",
        low_cpu_mem_usage=True,
        attn_implementation='sdpa',
        trust_remote_code=True
    )
    model.config.use_cache = False
    # Align model config with tokenizer tokens to avoid warnings
    if hasattr(model.config, 'pad_token_id'):
        model.config.pad_token_id = pad_token_id
    if hasattr(model.config, 'eos_token_id'):
        model.config.eos_token_id = eos_token_id
    if hasattr(model.config, 'bos_token_id') and tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map=None if use_ddp else "auto",
        trust_remote_code=True
    )
    ref_model.config.use_cache = False
    # Align reference model config with tokenizer tokens
    if hasattr(ref_model.config, 'pad_token_id'):
        ref_model.config.pad_token_id = pad_token_id
    if hasattr(ref_model.config, 'eos_token_id'):
        ref_model.config.eos_token_id = eos_token_id
    if hasattr(ref_model.config, 'bos_token_id') and tokenizer.bos_token_id is not None:
        ref_model.config.bos_token_id = tokenizer.bos_token_id


    # =============================================================================
    # Training arguments
    # =============================================================================

    # Ensure DDP is initialized before creating DPOConfig (it checks world_size internally)
    if use_ddp and not dist.is_initialized():
        print("Warning: DDP environment variables detected but init_process_group not called. Re-initializing...")
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

    # Base training arguments (always required)
    base_args = {
        'per_device_train_batch_size': TRAINING_CONFIG['per_device_train_batch_size'],
        'gradient_accumulation_steps': TRAINING_CONFIG['gradient_accumulation_steps'],
        'gradient_checkpointing': True,
        'learning_rate': TRAINING_CONFIG['learning_rate'],
        'lr_scheduler_type': "cosine",
        'num_train_epochs': 1,
        'save_strategy': "epoch",
        'logging_steps': 100,
        'output_dir': MODEL_CONFIG['new_model'],
        'optim': "paged_adamw_32bit",
        'warmup_steps': 50,
        'bf16': torch.cuda.is_bf16_supported(),
        'beta': 0.1,
        'max_prompt_length': TRAINING_CONFIG['max_prompt_length'],
        'max_length': TRAINING_CONFIG['max_length'],
        'report_to': "wandb" if (not use_ddp or rank == 0) else None,
        'dataloader_pin_memory': True,
        'dataloader_num_workers': min(4, os.cpu_count() or 4),
        'dataloader_drop_last': True,
        'remove_unused_columns': False,
    }

    # DDP-specific arguments (only when DDP is enabled)
    if use_ddp:
        base_args.update({
            'ddp_backend': "nccl",
            'ddp_find_unused_parameters': False,
            'local_rank': local_rank,
            'ddp_timeout': 1800,
        })

    training_args = DPOConfig(**base_args)

    peft_config = LoraConfig(
        r=LORA_CONFIG['r'],
        lora_alpha=LORA_CONFIG['alpha'],
        lora_dropout=LORA_CONFIG['dropout'],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
    )

    dpo_trainer = DPOTrainer(
        model,
        ref_model=None, #ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config
    )


    # =============================================================================
    # KL-divergence logging
    # =============================================================================

    sample_texts = []
    for i in range(min(100, len(eval_dataset))):  # Use 100 samples 
        sample_texts.append(eval_dataset[i]['prompt'])

    class KLDivergenceCallback(TrainerCallback):
        def __init__(self, reference_model, tokenizer, sample_texts):
            self.reference_model = reference_model
            self.tokenizer = tokenizer
            self.sample_texts = sample_texts
            
        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            
            if model is not None:
                
                kl_div = calculate_kl_divergence(
                    policy_model=model,
                    reference_model=self.reference_model,
                    tokenizer=self.tokenizer,
                    sample_texts=self.sample_texts
                )
                
                # wandb.log({
                #     "epoch": state.epoch,
                #     "kl_divergence": kl_div
                # })
                
                print(f"Epoch {state.epoch}: KL-divergence = {kl_div:.4f}")


    # =============================================================================
    # Training
    # =============================================================================

    print("Starting DPO training...")
    new_model = MODEL_CONFIG['new_model']

    if use_ddp:
        print(f"Using DDP with {world_size} processes for training")
        torch.cuda.empty_cache()
        gc.collect()
    elif torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        torch.cuda.empty_cache()
        gc.collect()

    if not use_ddp or rank == 0:
        kl_callback = KLDivergenceCallback(ref_model, tokenizer, sample_texts)
        dpo_trainer.add_callback(kl_callback)

    dpo_trainer.train()

    # Save artifacts
    if not use_ddp or rank == 0:
        checkpoint_dir = f"/ocean/projects/cis250219p/slee37/final_checkpoint/{LLM_model}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        dpo_trainer.model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"Model checkpoint saved to: {checkpoint_dir}")
    
        # Wait for all processes to finish saving
        if use_ddp:
            dist.barrier()

    del dpo_trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    # Reload model in FP16 (instead of NF4)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Align tokenizer and model config
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(base_model.config, 'pad_token_id'):
        base_model.config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if hasattr(base_model.config, 'eos_token_id'):
        base_model.config.eos_token_id = tokenizer.eos_token_id

    # Merge base model with the adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model = model.merge_and_unload()

    # Save model and tokenizer
    model.save_pretrained(new_model)
    tokenizer.save_pretrained(new_model)

    # Push them to the HF Hub (commented out to avoid permission issues)
    # model.push_to_hub(push_new_model, use_temp_dir=False, token=hf_token)
    # tokenizer.push_to_hub(push_new_model, use_temp_dir=False, token=hf_token)
    print(f"Model saved locally at: {new_model}")
    print("Hub upload disabled to avoid permission issues.")


    # =============================================================================
    # Evaluate Base Model (DPO)
    # =============================================================================
    messages = [{"role": "user", "content": "What is a Large Language Model?"}]

    tokenizer = AutoTokenizer.from_pretrained(new_model)
    model = AutoModelForCausalLM.from_pretrained(
        new_model,
        dtype=torch.float16,
        device_map="auto"
    )
    # Align tokenizer and model config
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model.config, 'pad_token_id'):
        model.config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if hasattr(model.config, 'eos_token_id'):
        model.config.eos_token_id = tokenizer.eos_token_id
    print(f"Successfully loaded model from: {new_model}")

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    outputs = pipeline(
        prompt,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        max_length=200,
    )

    toxicity_dataset = load_dataset("Anthropic/hh-rlhf")
    toxic_prompt_list = toxicity_dataset['test'].select(range(50))

    def map_initial_prompts(sample):
        return {"prompt": sample["chosen"].split("Assistant:")[0]}

    toxic_prompt_list = toxic_prompt_list.map(map_initial_prompts)


    def generate_output_from_prompt(sample, pipe):
        messages = [{"role": "user", "content": sample["prompt"].strip()}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        outputs = pipe(
            prompt,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            max_length=200,
        )

        return outputs[0]["generated_text"]

    base_model_generations = []
    for toxic_prompt in tqdm(toxic_prompt_list, desc=f"Generating for base model generation"):
        output = generate_output_from_prompt(toxic_prompt, pipeline)
        base_model_generations.append(output)

    base_model_generations_only_completions = []
    for generation in base_model_generations:
        base_model_generations_only_completions.append(generation.split("[/INST]")[-1])
        
    detoxify_model = Detoxify('original')

    toxicity_scores = []
    for text in base_model_generations_only_completions:
        result = detoxify_model.predict(text)
        toxicity_scores.append(result['toxicity'])

    toxicity_score = np.mean(toxicity_scores)
    print(f"Base Model(DPO) toxicity score: {toxicity_score:.4f}")
        

    del model, tokenizer, pipeline
    gc.collect()
    torch.cuda.empty_cache()
        

    # =============================================================================
    # Evaluate Original Model
    # =============================================================================

    orig_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        dtype=torch.float16
    )

    orig_model.config.use_cache = True

    message = [
        {"role": "user", "content": "What is a Large Language Model?"}
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Align tokenizer and model config
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(orig_model.config, 'pad_token_id'):
        orig_model.config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if hasattr(orig_model.config, 'eos_token_id'):
        orig_model.config.eos_token_id = tokenizer.eos_token_id
    prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)

    orig_pipeline = transformers.pipeline(
        "text-generation",
        model=orig_model,
        tokenizer=tokenizer
    )

    sequences = orig_pipeline(
        prompt,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        max_length=200,
    )

    orig_model_generations = []
    for toxic_prompt in tqdm(toxic_prompt_list):
        output = generate_output_from_prompt(toxic_prompt, orig_pipeline)
        orig_model_generations.append(output)

    orig_model_generations_only_completions = []
    for generation in orig_model_generations:
        orig_model_generations_only_completions.append(generation.split("[/INST]")[-1])

    detoxify_model = Detoxify('original')

    toxicity_scores = []
    for text in orig_model_generations_only_completions:
        result = detoxify_model.predict(text)
        toxicity_scores.append(result['toxicity'])

    ori_toxicity_score = np.mean(toxicity_scores)
    print(f"Original Model toxicity score: {ori_toxicity_score:.4f}")
        


    # =============================================================================
    # Evaluate DPO Model trained in 4bit
    # =============================================================================
    print("Loading pre-trained DPO model for evaluation...")


    dpo_model = AutoModelForCausalLM.from_pretrained(
        my_dpo_model,
        dtype=torch.float16,
        load_in_4bit=True
    )
    dpo_model.config.use_cache = True

    message = [
        {"role": "user", "content": "What is a Large Language Model?"}
    ]
    tokenizer = AutoTokenizer.from_pretrained(my_dpo_model)
    # Align tokenizer and model config
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(dpo_model.config, 'pad_token_id'):
        dpo_model.config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if hasattr(dpo_model.config, 'eos_token_id'):
        dpo_model.config.eos_token_id = tokenizer.eos_token_id
    prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)

    dpo_pipeline = transformers.pipeline(
        "text-generation",
        model=dpo_model,
        tokenizer=tokenizer
    )

    sequences = dpo_pipeline(
        prompt,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        max_length=200,
    )

    dpo_model_generations = []
    for toxic_prompt in tqdm(toxic_prompt_list):
        output = generate_output_from_prompt(toxic_prompt, dpo_pipeline)
        dpo_model_generations.append(output)

    dpo_model_generations_only_completions = []

    for generation in dpo_model_generations:
        dpo_model_generations_only_completions.append(generation.split("[/INST]")[-1])

    detoxify_model = Detoxify('original')

    toxicity_scores = []
    for text in dpo_model_generations_only_completions:
        result = detoxify_model.predict(text)
        toxicity_scores.append(result['toxicity'])

    dpo_toxicity_score = np.mean(toxicity_scores)
    print(f"DPO Model toxicity score: {dpo_toxicity_score:.4f}")
        

    print(f"\n=== TOXICITY COMPARISON ===")
    print(f"Original Model: {ori_toxicity_score:.4f}")
    print(f"Base Model (DPO): {toxicity_score:.4f}")
    print(f"Pre-trained DPO Model: {dpo_toxicity_score:.4f}")

    wandb.finish()



if __name__ == "__main__":
    main()