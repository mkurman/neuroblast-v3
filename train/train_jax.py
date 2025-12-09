import sys
import os

sys.path.append("../")

from transformers import AutoTokenizer, logging
from datasets import load_dataset, concatenate_datasets
from datetime import datetime
from typing import Union, Optional
import random
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from flax.training import train_state
from flax.core.frozen_dict import FrozenDict
from flax import jax_utils
import optax
import orbax.checkpoint as ocp
import warnings
import wandb
from functools import partial
from tqdm import tqdm
import math

from model.jax.configuration_neuroblast_jax import NeuroBLASTConfig
from model.jax.modeling_neuroblast_jax import NeuroBLASTForCausalLM

def checkpoint_barrier(tag: str, step: int):
    multihost_utils.sync_global_devices(f"{tag}_step_{int(step)}")

print(f"DEBUG: JAX_COORDINATOR_ADDRESS={os.environ.get('JAX_COORDINATOR_ADDRESS')}")
print(f"DEBUG: JAX_PROCESS_COUNT={os.environ.get('JAX_PROCESS_COUNT')}")
print(f"DEBUG: JAX_PROCESS_INDEX={os.environ.get('JAX_PROCESS_INDEX')}")

coordinator_address = os.environ.get('JAX_COORDINATOR_ADDRESS')
num_processes = os.environ.get('JAX_PROCESS_COUNT')
process_id = os.environ.get('JAX_PROCESS_INDEX')

num_processes = int(num_processes) if num_processes else 1
process_id = int(process_id) if process_id else 0
coordinator_address = coordinator_address if coordinator_address else "localhost:1234"

try:
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id
    )
    print(f"✅ JAX process: {jax.process_index()}/{jax.process_count()} initialized successfully.")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
except Exception as e:
    print(f"ERROR: Failed to initialize JAX distributed training: {e}")
    exit(1)

warnings.filterwarnings("ignore")


def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def filter_item(example, length: int = 100):
    return len(example["text"].strip()) > length


def filter_synth(x):
    return "query" in x and "synthetic_reasoning" in x and "synthetic_answer" in x


def map_synth(x):
    return {
        "text": "user\n"
        + x["query"]
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
        + "\n<think>\n"
        + x["synthetic_reasoning"]
        + "\n</think>\n\n"
        + x["synthetic_answer"]
        + "<|im_end|>",
    }

def map_synth_chat(x):
    return [
        {
            "role": "user", 
            "content": x["query"]
        },
        {
            "role": "assistant",
            "content": "<think>\n"
            + x["synthetic_reasoning"]
            + "\n</think>\n\n"
            + x["synthetic_answer"]
        }
    ]


class TrainState(train_state.TrainState):
    dropout_rng: jax.random.PRNGKey


def create_learning_rate_fn(warmup_steps: int, max_steps: int, base_learning_rate: float, num_decay_steps: int = 20000):
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=base_learning_rate, transition_steps=warmup_steps)
    stable_steps = max_steps - warmup_steps - num_decay_steps
    if stable_steps < 0: stable_steps = 0
    stable_fn = optax.constant_schedule(base_learning_rate)
    decay_fn = optax.linear_schedule(init_value=base_learning_rate, end_value=0.0, transition_steps=num_decay_steps)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, stable_fn, decay_fn], boundaries=[warmup_steps, warmup_steps + stable_steps])
    return schedule_fn


def create_train_state(rng, config, max_length, learning_rate, num_steps, warmup_steps, weight_decay, decay_steps):
    model = NeuroBLASTForCausalLM(config, dtype=jnp.bfloat16)
    input_shape = (1, max_length)
    params = model.init_weights(rng, input_shape)
    
    lr_schedule = create_learning_rate_fn(warmup_steps, num_steps, learning_rate, decay_steps)
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.999, eps=1e-8, weight_decay=weight_decay)
    )
    
    state = TrainState.create(
        apply_fn=model.__call__,
        params=params,
        tx=optimizer,
        dropout_rng=jax.random.PRNGKey(0),
    )
    return state, model


def compute_loss(logits, labels, attention_mask=None, label_smoothing=0.0):
    """
    Compute cross-entropy loss with proper padding token masking.
    
    Args:
        logits: (B, L, vocab_size) - model predictions
        labels: (B, L) - target token ids (includes padding)
        attention_mask: (B, L) - 1 for real tokens, 0 for padding
        label_smoothing: float - label smoothing factor
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    
    if label_smoothing > 0.0:
        vocab_size = shift_logits.shape[-1]
        one_hot_labels = jax.nn.one_hot(shift_labels, vocab_size)
        smoothed_labels = optax.smooth_labels(one_hot_labels, label_smoothing)
        loss = optax.softmax_cross_entropy(shift_logits, smoothed_labels)
    else:
        loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels)
    
    if attention_mask is not None:
        # Shift the attention mask to align with shifted labels
        shift_mask = attention_mask[:, 1:].astype(jnp.float32)
        # Multiply loss by mask (zeros out padding positions)
        loss = loss * shift_mask
        # Average only over non-padding tokens
        loss = jnp.sum(loss) / jnp.maximum(jnp.sum(shift_mask), 1.0)
    else:
        loss = jnp.mean(loss)
    return loss



def train_step_accumulated(state, batch_sequences, dropout_rng, label_smoothing=0.0):
    """
    Args:
        state: TrainState
        batch_sequences: Dictionary of arrays with shape [accum_steps, batch_size, seq_len]
        dropout_rng: RNG Key
    """
    
    # 1. Define the step for a single micro-batch
    def mini_batch_step(carry, mini_batch):
        accum_grads, accum_loss, rng = carry
        
        # Split RNG for this micro-step
        step_rng, next_rng = jax.random.split(rng)
        
        def loss_fn(params):
            logits = state.apply_fn(
                input_ids=mini_batch['input_ids'],
                attention_mask=mini_batch.get('attention_mask'),
                params=params,
                dropout_rng=step_rng,
                train=True,
                return_dict=True,
            ).logits
            
            loss = compute_loss(
                logits=logits,
                labels=mini_batch['input_ids'],
                attention_mask=mini_batch.get('attention_mask'),
                label_smoothing=label_smoothing
            )
            
            # Normalize loss immediately by total accumulation steps
            # batch_sequences['input_ids'].shape[0] is the accumulation count
            return loss / batch_sequences['input_ids'].shape[0]

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        
        # Accumulate gradients (JAX optimizes this to be in-place)
        new_accum_grads = jax.tree.map(lambda x, y: x + y, accum_grads, grads)
        new_accum_loss = accum_loss + loss
        
        return (new_accum_grads, new_accum_loss, next_rng), None

    # 2. Initialize accumulator with zeros
    zero_grads = jax.tree.map(jnp.zeros_like, state.params)
    initial_carry = (zero_grads, 0.0, dropout_rng)

    # 3. Use lax.scan to loop over the accumulation dimension efficiently
    # This keeps memory CONSTANT regardless of how many accumulation steps you use
    (final_grads, total_loss, final_rng), _ = jax.lax.scan(
        mini_batch_step, 
        initial_carry, 
        batch_sequences
    )
    
    # 4. Apply the accumulated gradients once
    new_state = state.apply_gradients(grads=final_grads)
    
    # Update state RNG
    new_state = new_state.replace(dropout_rng=final_rng)
    
    return new_state, total_loss

def eval_step(state, batch):
    logits = state.apply_fn(
        input_ids=batch['input_ids'],
        attention_mask=batch.get('attention_mask'),
        params=state.params,
        train=False,
        return_dict=True,
    ).logits
    loss = compute_loss(logits, batch['input_ids'], batch.get('attention_mask'))
    return {'loss': loss}


# donate_argnums=(0,) is CRITICAL. It allows JAX to overwrite the input state memory 
# with the output state memory, preventing memory doubling.
p_train_step = jax.pmap(
    train_step_accumulated, 
    axis_name='batch', 
    donate_argnums=(0,),
    static_broadcasted_argnums=(3,)
)
p_eval_step = jax.pmap(eval_step, axis_name='batch')


def unreplicate_state_to_host(state):
    """Take the first device copy and move it to host/NumPy for checkpointing."""
    single = jax.tree.map(lambda x: x[0], state)
    return jax.device_get(single)


def prepare_macro_batch(batch_data, num_devices, accum_iter):
    """
    Reshapes flat batch data into [num_devices, accum_iter, batch_per_device, seq_len]
    """
    def reshape_fn(x):
        if x is None: return None
        # x shape: [total_items, seq_len]
        # target: [num_devices, accum_iter, batch_per_device, seq_len]
        batch_per_device = x.shape[0] // (num_devices * accum_iter)
        return x.reshape((num_devices, accum_iter, batch_per_device) + x.shape[1:])
    
    return {k: reshape_fn(v) for k, v in batch_data.items()}


def prepare_eval_batch(batch_data, num_devices):
    """
    Reshapes flat eval batch into [num_devices, batch_per_device, seq_len]
    """
    def reshape_fn(x):
        if x is None: return None
        batch_per_device = x.shape[0] // num_devices
        return x.reshape((num_devices, batch_per_device) + x.shape[1:])
    
    return {k: reshape_fn(v) for k, v in batch_data.items()}


def train(
    model_name: str,
    checkpoint: str,
    trainer_checkpoint: str,
    tokenizer_name: str,
    dataset_train: str,
    dataset_validation: Optional[str],
    max_length: int = 512,
    batch_size: int = 32, # Batch size per device per micro-step
    accumulation_iter: int = 128,
    epochs: int = 1,
    lr: float = 5e-5,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    logging_dir: str = "./logs",
    output_dir: str = "./results",
    save_steps: int = 500,
    train_test_split: Union[int, float] = 0.1,
    seed: int = 42,
    reset_training_progress: bool = False,
    label_smoothing: float = 0.0,
    max_samples: int = None,
    **kwargs,
):
    print(f"Training {model_name} on {dataset_train} with {tokenizer_name} tokenizer.")

    set_all_seeds(seed)
    logger = logging.get_logger(__name__)

    print(f"Loading tokenizer {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    if jax.process_index() == 0:
        print("Tokenizer loaded.", flush=True)

    num_devices = jax.local_device_count()
    global_devices = jax.device_count()
    
    if jax.process_index() == 0:
        wandb.init(
            project="frankie_big_context",
            name=f"{model_name}_{datetime.now().strftime('%Y-%m-%d %H_%M_%S')}",
        )

    config = NeuroBLASTConfig(
        vocab_size=len(tokenizer),
        hidden_size=512,
        intermediate_size=3072,
        num_associative_layers=32,
        num_sensory_layers=24,
        num_motor_layers=16,
        num_hidden_layers=72,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=True,
        attention_dropout=0.05,
        dropout=0.05,
    )

    config.save_pretrained(output_dir)


    print("Loading model...")
    
    items_per_optimizer_step = batch_size * num_devices * accumulation_iter
    
    train_samples = max_samples if max_samples else 10000000 # Approximation for infinite stream
    max_steps = train_samples // items_per_optimizer_step * epochs
    
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    
    if trainer_checkpoint is not None:
        print("Initializing abstract model state to save memory...")
        def init_state_fn():
            state, _ = create_train_state(
                init_rng, config, max_length, lr, max_steps, warmup_steps, weight_decay, int(train_samples * 0.1)
            )
            return state
        abstract_state = jax.eval_shape(init_state_fn)
        state = abstract_state
    else:
        print("Initializing random model state...")
        try:
            cpu_device = jax.devices('cpu')[0]
        except:
            cpu_device = jax.local_devices(backend='cpu')[0] if jax.local_devices(backend='cpu') else None

        if cpu_device:
            with jax.default_device(cpu_device):
                state, model = create_train_state(
                    init_rng, config, max_length, lr, max_steps, warmup_steps, weight_decay, int(train_samples * 0.1)
                )
        else:
            state, model = create_train_state(
                init_rng, config, max_length, lr, max_steps, warmup_steps, weight_decay, int(train_samples * 0.1)
            )

    if trainer_checkpoint is not None:
        print(f"Attempting to restore checkpoint from {trainer_checkpoint}...")
        try:
            checkpointer = ocp.StandardCheckpointer()
            load_options = ocp.CheckpointManagerOptions(max_to_keep=None, create=False)
            load_manager = ocp.CheckpointManager(os.path.abspath(trainer_checkpoint), options=load_options)
            
            latest_step = load_manager.latest_step()
            if latest_step is not None:
                print(f"Found checkpoint at step {latest_step}. Restoring...")
                
                restored_state = load_manager.restore(latest_step, args=ocp.args.StandardRestore(state))
                
                if restored_state is None:
                    restored_state = checkpointer.restore(
                        trainer_checkpoint, item=state
                    )
                    if restored_state is None:
                         restored_state = checkpointer.restore(trainer_checkpoint)
                         if isinstance(restored_state, dict):
                             from flax import serialization
                             pass
                    
                    if restored_state is None:
                        raise ValueError(f"No checkpoints found in {trainer_checkpoint}")

                if restored_state is not None:
                    state = restored_state
                    print(f"Successfully restored state from step {latest_step}.")
                    
                    if reset_training_progress:
                        print(f"CPT: Resetting training progress (step=0) while keeping optimizer states.")
                        state = state.replace(step=0)
                        
                        def reset_count(leaf):
                            if isinstance(leaf, jnp.ndarray) and leaf.ndim == 0 and leaf.dtype == jnp.int32:
                                return jnp.zeros_like(leaf)
                            return leaf
                            
                        new_opt_state = jax.tree.map(reset_count, state.opt_state)
                        state = state.replace(opt_state=new_opt_state)
                else:
                    print("Warning: Restore returned None. Starting from scratch.")
                    state, model = create_train_state(
                        init_rng, config, max_length, lr, max_steps, warmup_steps, weight_decay, int(train_samples * 0.1)
                    )
            else:
                print(f"No checkpoint found in {trainer_checkpoint}. Starting from scratch.")
                state, model = create_train_state(
                    init_rng, config, max_length, lr, max_steps, warmup_steps, weight_decay, int(train_samples * 0.1)
                )
        except Exception as e:
            print("Starting from scratch.")
            import traceback
            traceback.print_exc()
            state, model = create_train_state(
                init_rng, config, max_length, lr, max_steps, warmup_steps, weight_decay, int(train_samples * 0.1)
            )

    state = jax_utils.replicate(state)
    if jax.process_index() == 0:
        print("Model loaded and replicated across devices.", flush=True)

    if jax.process_index() == 0:
        print("Loading dataset...", flush=True)
    synth = load_dataset(
        "PleIAs/SYNTH",
        streaming=True,
        split="train",
        data_files=["synth_*.parquet"],
        cache_dir="hf_cache",
    )

    dataset = (
        synth
        .skip(100)
        .shuffle(seed=seed, buffer_size=10000)
        .filter(filter_synth)
        .map(lambda x: {"text": tokenizer.apply_chat_template(map_synth_chat(x), tokenize=False)})
        .filter(filter_item)
    )
    if jax.process_index() == 0:
        print("Dataset pipeline built.", flush=True)

    eval_text_cache = None

    if trainer_checkpoint is None:
        output_dir = os.path.join(
            os.getcwd(), output_dir, model_name,
            datetime.now().strftime("%Y-%m-%d %H_%M_%S"),
        )
    else:
        output_dir = "/".join(trainer_checkpoint.split("/")[:-1])
    
    os.makedirs(output_dir, exist_ok=True)
    
    abs_output_dir = os.path.abspath(output_dir)
    options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpointer = ocp.StandardCheckpointer()
    manager = ocp.CheckpointManager(abs_output_dir, checkpointer, options=options)

    print(f"Output directory: {output_dir}")
    print(f"Optimizer Steps: {max_steps}")
    print(f"Batch size per device: {batch_size}")
    print(f"Accumulation steps: {accumulation_iter}")
    print(f"Items processed per optimizer step: {items_per_optimizer_step}")
    
    print("Starting training...")
    step = 0
    num_input_tokens_seen = 0  # Track total non-padding tokens processed
    dataset_iter = iter(dataset)
    if jax.process_index() == 0:
        print("Dataset iterator created, entering training loop.", flush=True)
    
    global lr_schedule
    lr_schedule = create_learning_rate_fn(warmup_steps, max_steps, lr, 20000)
    
    with tqdm(total=max_steps, desc="Training") as pbar:
        while step < max_steps:
            try:
                batch_texts = []
                
                for _ in range(items_per_optimizer_step):
                    try:
                        example = next(dataset_iter)
                        batch_texts.append(example['text'])
                    except StopIteration:
                        break
                if jax.process_index() == 0 and step == 0:
                    print(f"Fetched {len(batch_texts)} samples for first step.", flush=True)
                
                if len(batch_texts) < items_per_optimizer_step:
                    print(f"Not enough data for full batch ({len(batch_texts)} < {items_per_optimizer_step}), stopping.")
                    break

                batch_encoding = tokenizer(
                    batch_texts,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='np',
                )
                
                flat_batch = {
                    'input_ids': batch_encoding['input_ids'],
                    'attention_mask': batch_encoding['attention_mask']
                }
                
                macro_batch_np = prepare_macro_batch(flat_batch, num_devices, accumulation_iter)
                
                macro_batch_jax = {
                    'input_ids': jnp.array(macro_batch_np['input_ids']),
                    'attention_mask': jnp.array(macro_batch_np['attention_mask'])
                }
                
                rng, dropout_rng = jax.random.split(rng)
                dropout_rngs = jax.random.split(dropout_rng, num_devices)
                
                state, loss_value = p_train_step(state, macro_batch_jax, dropout_rngs, label_smoothing)
                
                step += 1
                
                tokens_this_step = int(jnp.sum(macro_batch_jax['attention_mask']))
                num_input_tokens_seen += tokens_this_step
                
                pbar.update(1)
                
                if step % 1 == 0:
                    avg_loss = float(loss_value[0])
                    
                    current_lr_step = int(step)
                    lr_value = float(lr_schedule(current_lr_step))
                    
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr_value:.2e}', 'tokens': f'{num_input_tokens_seen:,}'})
                    
                    if jax.process_index() == 0:
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/learning_rate': lr_value,
                            'train/step': step,
                            'train/num_input_tokens_seen': num_input_tokens_seen,
                        })

                if step > 0 and step % save_steps == 0:
                    if jax.process_index() == 0:
                        print(f"\nSaving checkpoint at step {step}")
                        
                    state_to_save = jax.device_get(jax_utils.unreplicate(state))
                    
                    manager.save(
                        step, 
                        args=ocp.args.StandardSave(state_to_save)
                    )
                    if jax.process_index() == 0:
                        print(f"Checkpoint saved to {output_dir}")

                    if jax.process_index() == 0:
                        if eval_text_cache is None:
                            try:
                                eval_synth = load_dataset(
                                    "PleIAs/SYNTH",
                                    streaming=True,
                                    split="train",
                                    data_files=["synth_*.parquet"],
                                    cache_dir="hf_cache",
                                )
                                eval_dataset = (
                                    eval_synth
                                    .take(num_devices * batch_size)
                                    .shuffle(seed=seed + 1, buffer_size=10000)
                                    .filter(filter_synth)
                                    .map(map_synth)
                                    .filter(filter_item)
                                )
                                eval_text_cache = [ex["text"] for ex in eval_dataset]
                            except Exception as e:
                                print(f"Eval dataset unavailable, skipping eval: {e}")
                                eval_text_cache = []

                        if len(eval_text_cache) == num_devices * batch_size:
                            eval_encoding = tokenizer(
                                eval_text_cache,
                                max_length=max_length,
                                padding='max_length',
                                truncation=True,
                                return_tensors='np',
                            )
                            eval_flat = {
                                'input_ids': eval_encoding['input_ids'],
                                'attention_mask': eval_encoding['attention_mask']
                            }
                            eval_batch_np = prepare_eval_batch(eval_flat, num_devices)
                            eval_batch_jax = {
                                'input_ids': jnp.array(eval_batch_np['input_ids']),
                                'attention_mask': jnp.array(eval_batch_np['attention_mask'])
                            }
                            eval_metrics = p_eval_step(state, eval_batch_jax)
                            eval_loss = float(jnp.mean(eval_metrics['loss']))
                            print(f"Eval loss at step {step}: {eval_loss:.4f}")
                            wandb.log({'eval/loss': eval_loss, 'eval/step': step})

            except Exception as e:
                print(f"\nError during training: {e}")
                import traceback
                traceback.print_exc()
                break

    print("Training complete!")
    
    if jax.process_index() == 0 and manager is not None:
        state_to_save = unreplicate_state_to_host(state)
        
        manager.save(step, state_to_save)
        manager.wait_until_finished()
        print(f"Final checkpoint saved to {output_dir}")
        wandb.finish()


def main():
    print("Training model (Optimized).")
    train(
        model_name="neuroblast_jax",
        checkpoint=None,
        trainer_checkpoint=None,
        dataset_train=None,
        dataset_validation=None,
        max_length=1280,
        tokenizer_name="PleIAs/Baguettotron",
        batch_size=2,   
        accumulation_iter=32,
        epochs=1,
        lr=4e-3,
        weight_decay=0.0,
        logging_dir="logs",
        output_dir="results",
        save_steps=1000,
        seed=3407,
        max_samples=100000000,
        reset_training_progress=True, # Set to True for CPT
        warmup_steps=1000,
        label_smoothing=0.1,
    )


if __name__ == "__main__":
    main()
