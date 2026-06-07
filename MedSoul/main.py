"""
Main pipeline controller for MedSoul
Runs the complete pipeline: LLM labeling -> MAE pretraining -> WSL training -> Confident Learning
"""
import os
import sys
import yaml
import subprocess
import shutil
from pathlib import Path
from datetime import datetime


def list_existing_experiments():
    """List all existing experiments in outputs/ directory"""
    outputs_dir = Path('outputs')
    if not outputs_dir.exists():
        return []
    
    experiments = []
    for item in outputs_dir.iterdir():
        if item.is_dir() and (item / 'config.yaml').exists():
            experiments.append(item.name)
    
    return sorted(experiments)


def select_experiment_mode():
    """Ask user if they want to create new or continue existing experiment"""
    print("\n" + "="*70)
    print("  Experiment Mode Selection")
    print("="*70)
    
    existing_exps = list_existing_experiments()
    
    print("\nOptions:")
    print("  1. Create new experiment")
    if existing_exps:
        print("  2. Continue existing experiment")
        print("  3. Cancel")
    else:
        print("  2. Cancel")
    
    while True:
        if existing_exps:
            choice = input("\nYour choice (1/2/3): ").strip()
        else:
            choice = input("\nYour choice (1/2): ").strip()
        
        if choice == '1':
            return 'new', None
        elif choice == '2' and existing_exps:
            # Show list of experiments
            print("\n" + "="*70)
            print("  Available Experiments")
            print("="*70)
            for i, exp_name in enumerate(existing_exps, 1):
                print(f"  {i}. {exp_name}")
            
            while True:
                exp_choice = input(f"\nSelect experiment (1-{len(existing_exps)}): ").strip()
                try:
                    idx = int(exp_choice) - 1
                    if 0 <= idx < len(existing_exps):
                        return 'continue', existing_exps[idx]
                    else:
                        print(f"Please enter a number between 1 and {len(existing_exps)}")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        elif choice == '2' or choice == '3':
            print("Cancelled.")
            sys.exit(0)
        else:
            print("Invalid choice.")


def get_experiment_name():
    """Get experiment name from user"""
    print("\n" + "="*70)
    print("  New Experiment Configuration")
    print("="*70)
    
    while True:
        exp_name = input("\nEnter experiment name (e.g., 'baseline', 'with_cl', 'test_01'): ").strip()
        if exp_name:
            # Sanitize name
            exp_name = exp_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            
            exp_dir = Path('outputs') / exp_name
            if exp_dir.exists():
                print(f"\n[WARN] Experiment '{exp_name}' already exists!")
                choice = input("Choose: (o)verwrite, (r)ename, (c)ancel: ").strip().lower()
                if choice == 'o':
                    confirm = input(f"Confirm overwrite '{exp_name}'? (yes/no): ").strip().lower()
                    if confirm == 'yes':
                        shutil.rmtree(exp_dir)
                        print(f"[INFO] Removed existing experiment: {exp_name}")
                        return exp_name, exp_dir
                elif choice == 'r':
                    continue
                else:
                    print("Cancelled.")
                    sys.exit(0)
            else:
                return exp_name, exp_dir
        else:
            print("[ERROR] Experiment name cannot be empty")


def create_experiment_config(base_config, exp_dir, iteration=0):
    """
    Create experiment-specific config with updated paths
    
    Args:
        base_config: Base configuration
        exp_dir: Experiment directory
        iteration: Iteration number (0 for initial, 1+ for CL iterations)
    """
    exp_config = base_config.copy()
    
    # Update paths to be relative to experiment directory
    # This allows scripts to automatically resolve them correctly
    if iteration > 0:
        # For iterations after CL, use cleaned labels
        exp_config['llm']['cache_file'] = 'confident_learning/pseudo_labels_cleaned.json'
        exp_config['mae']['checkpoint_dir'] = f'mae_pretrain_iter{iteration}'
        exp_config['wsl']['checkpoint_dir'] = f'wsl_train_iter{iteration}'
        # Don't run CL again in iterations
        exp_config['confident_learning']['enabled'] = False
    else:
        # Initial run - use relative paths
        exp_config['llm']['cache_file'] = 'pseudo_labels.json'
        exp_config['mae']['checkpoint_dir'] = 'mae_pretrain'
        exp_config['wsl']['checkpoint_dir'] = 'wsl_train'
    
    exp_config['confident_learning']['output_dir'] = 'confident_learning'
    exp_config['pipeline']['output_dir'] = str(exp_dir)  # This is the base path
    
    # Create necessary directories (relative to exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / exp_config['mae']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    (exp_dir / exp_config['wsl']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    (exp_dir / exp_config['confident_learning']['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Save experiment config
    config_filename = f'config_iter{iteration}.yaml' if iteration > 0 else 'config.yaml'
    exp_config_path = exp_dir / config_filename
    with open(exp_config_path, 'w') as f:
        yaml.dump(exp_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n[INFO] Experiment config saved: {exp_config_path}")
    
    return exp_config, exp_config_path


def load_experiment_config(exp_name):
    """Load configuration from existing experiment"""
    exp_dir = Path('outputs') / exp_name
    config_path = exp_dir / 'config.yaml'
    
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n[INFO] Loaded experiment: {exp_name}")
    print(f"[INFO] Config path: {config_path}")
    
    return config, config_path, exp_dir


def run_command(cmd, description):
    """Run a command and handle errors"""
    print("\n" + "="*70)
    print(f"  {description}")
    print("="*70)
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed")
        print(f"Command failed with return code {result.returncode}")
        return False
    
    print(f"\n[OK] {description} completed successfully")
    return True


def check_phase_completed(exp_config, phase_name):
    """
    Check if a training phase has been completed
    
    For MAE and WSL, checks:
    1. If best model files exist (primary check)
    2. If all epochs have been completed based on epoch checkpoint files
    """
    # Get base experiment directory
    base_dir = Path(exp_config.get('pipeline', {}).get('output_dir', 'outputs'))
    
    if phase_name == 'mae':
        # Check for MAE encoder checkpoint
        mae_dir = base_dir / exp_config.get('mae', {}).get('checkpoint_dir', 'mae_pretrain')
        encoder_path = mae_dir / 'encoder_best.pth'
        mae_best_path = mae_dir / 'mae_best.pth'
        
        print(f"[DEBUG] Checking MAE at: {mae_dir.absolute()}")
        
        # Primary check: both best files exist
        if encoder_path.exists() and mae_best_path.exists():
            print(f"[DEBUG] Found MAE best models")
            return True
        
        # Secondary check: all epochs completed based on epoch files
        target_epochs = exp_config.get('mae', {}).get('epochs', 100)
        epoch_files = list(mae_dir.glob('mae_epoch_*.pth'))
        
        if epoch_files:
            max_epoch = 0
            for f in epoch_files:
                try:
                    epoch_num = int(f.stem.split('_')[-1])
                    max_epoch = max(max_epoch, epoch_num)
                except (ValueError, IndexError):
                    continue
            
            if max_epoch >= target_epochs:
                print(f"[DEBUG] MAE reached epoch {max_epoch}/{target_epochs} but best models missing")
                return True
        
        return False
    
    elif phase_name == 'wsl':
        # Check for WSL fine-tuned model
        wsl_dir = base_dir / exp_config.get('wsl', {}).get('checkpoint_dir', 'wsl_train')
        ft_best_path = wsl_dir / 'fine_tune_best.pth'
        
        print(f"[DEBUG] Checking WSL at: {wsl_dir.absolute()}")
        
        # Primary check: fine_tune_best exists
        if ft_best_path.exists():
            print(f"[DEBUG] Found WSL fine-tune best model")
            return True
        
        # Secondary check: all epochs completed for fine-tune
        target_epochs = exp_config.get('wsl', {}).get('fine_tune', {}).get('epochs', 100)
        epoch_files = list(wsl_dir.glob('fine_tune_epoch_*.pth'))
        
        if epoch_files:
            max_epoch = 0
            for f in epoch_files:
                try:
                    epoch_num = int(f.stem.split('_')[-1])
                    max_epoch = max(max_epoch, epoch_num)
                except (ValueError, IndexError):
                    continue
            
            if max_epoch >= target_epochs:
                print(f"[DEBUG] WSL fine-tune reached epoch {max_epoch}/{target_epochs} but best model missing")
                return True
        
        return False
    
    elif phase_name == 'confident_learning':
        # Check for cleaned labels
        cl_dir = base_dir / exp_config.get('confident_learning', {}).get('output_dir', 'confident_learning')
        cleaned_path = cl_dir / exp_config.get('confident_learning', {}).get('cleaned_labels_file', 'pseudo_labels_cleaned.json')
        print(f"[DEBUG] Checking CL at: {cleaned_path.absolute()}")
        return cleaned_path.exists()
    
    return False


def run_training_iteration(exp_config, exp_config_path, skip_phase1=True, skip_phase4=False, iteration=0):
    """
    Run training phases
    
    Args:
        exp_config: Experiment configuration
        exp_config_path: Path to config file
        skip_phase1: Whether to skip Phase 1 (label generation)
        skip_phase4: Whether to skip Phase 4 (Confident Learning)
        iteration: Iteration number for display
    """
    phases = exp_config.get('pipeline', {}).get('phases', {})
    
    iteration_suffix = f" (Iteration {iteration})" if iteration > 0 else ""
    
    # Phase 1: Generate pseudo labels (optional)
    if not skip_phase1 and phases.get('generate_labels', True):
        success = run_command(
            f"conda activate .\.conda && python generate_labels.py --config {exp_config_path}",
            f"Phase 1: Generating Pseudo Labels{iteration_suffix}"
        )
        if not success:
            print("\n[ERROR] Phase 1 failed. Exiting...")
            return False
    elif skip_phase1:
        print(f"\n[SKIP] Phase 1: Using existing pseudo labels{iteration_suffix}")
    
    # Phase 2: MAE pretraining
    if phases.get('train_mae', True):
        # Check if already completed (skip in continuation mode)
        if check_phase_completed(exp_config, 'mae'):
            print(f"\n[SKIP] Phase 2: MAE already trained (encoder_best.pth exists){iteration_suffix}")
        else:
            success = run_command(
                f"conda activate .\.conda && python train_mae.py --config {exp_config_path}",
                f"Phase 2: MAE Self-Supervised Pretraining{iteration_suffix}"
            )
            if not success:
                print("\n[ERROR] Phase 2 failed. Exiting...")
                return False
    else:
        print(f"\n[SKIP] Phase 2 disabled in config")
    
    # Phase 3: WSL training
    if phases.get('train_wsl', True):
        # Check if already completed (skip in continuation mode)
        if check_phase_completed(exp_config, 'wsl'):
            print(f"\n[SKIP] Phase 3: WSL already trained (fine_tune_best.pth exists){iteration_suffix}")
        else:
            success = run_command(
                f"conda activate .\.conda && python train_wsl.py --config {exp_config_path}",
                f"Phase 3: Weakly Supervised Learning{iteration_suffix}"
            )
            if not success:
                print("\n[ERROR] Phase 3 failed. Exiting...")
                return False
    else:
        print(f"\n[SKIP] Phase 3 disabled in config")
    
    # Phase 4: Confident Learning
    if not skip_phase4 and phases.get('confident_learning', True) and exp_config.get('confident_learning', {}).get('enabled', True):
        # Check if already completed (skip in continuation mode)
        if check_phase_completed(exp_config, 'confident_learning'):
            print(f"\n[SKIP] Phase 4: Confident Learning already completed (cleaned labels exist){iteration_suffix}")
        else:
            success = run_command(
                f"conda activate .\.conda && python confident_learning.py --config {exp_config_path}",
                f"Phase 4: Confident Learning (Label Cleaning){iteration_suffix}"
            )
            if not success:
                print("\n[ERROR] Phase 4 failed. Exiting...")
                return False
            
            print("\n" + "="*70)
            print("  Confident Learning Completed")
            print("="*70)
            print("\nCleaned labels saved to:")
            print(f"  {exp_config['confident_learning']['output_dir']}/pseudo_labels_cleaned.json")
            print("\nOriginal labels preserved at:")
            print(f"  {exp_config['llm']['cache_file']}")
            
            return True  # Indicate CL was run
    
    return False  # CL was not run


def detect_experiment_progress(exp_dir):
    """
    Detect which iteration the experiment is at and whether it's complete
    
    Returns:
        (last_completed_iteration, has_incomplete_iteration, incomplete_iteration_num)
    """
    exp_dir = Path(exp_dir)
    last_completed = -1
    incomplete_iteration = None
    
    # Check iteration 0
    if (exp_dir / 'config.yaml').exists():
        with open(exp_dir / 'config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        
        mae_done = check_phase_completed(cfg, 'mae')
        wsl_done = check_phase_completed(cfg, 'wsl')
        
        if mae_done and wsl_done:
            last_completed = 0
        elif mae_done or wsl_done:
            # Partially completed
            incomplete_iteration = 0
    
    # Check iterations 1+
    iteration = 1
    while (exp_dir / f'config_iter{iteration}.yaml').exists():
        with open(exp_dir / f'config_iter{iteration}.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
        
        mae_done = check_phase_completed(cfg, 'mae')
        wsl_done = check_phase_completed(cfg, 'wsl')
        
        if mae_done and wsl_done:
            last_completed = iteration
        elif mae_done or wsl_done:
            # Partially completed
            incomplete_iteration = iteration
            break
        else:
            # Not started
            break
        
        iteration += 1
    
    return last_completed, incomplete_iteration


def main():
    # Load base config
    config_path = Path('configs/config_mimic_jpg.yaml')
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    print("="*70)
    print("  MedSoul Pipeline")
    print("  Weakly Supervised Learning for Medical Image Classification")
    print("="*70)
    
    # Select experiment mode
    mode, existing_exp_name = select_experiment_mode()
    
    if mode == 'continue':
        # Continue existing experiment
        exp_config, exp_config_path, exp_dir = load_experiment_config(existing_exp_name)
        exp_name = existing_exp_name
        skip_phase1 = True  # Always skip when continuing
        
        print("\n[INFO] Continuing experiment from existing configuration")
        
        # Detect experiment progress
        last_completed, incomplete_iteration = detect_experiment_progress(exp_dir)
        
        print(f"\n[INFO] Experiment progress:")
        if last_completed >= 0:
            print(f"  ✓ Completed iterations: 0-{last_completed}")
        if incomplete_iteration is not None:
            print(f"  ⚠ Incomplete iteration: {incomplete_iteration}")
        
        # Resume from incomplete iteration or start next
        if incomplete_iteration is not None:
            print(f"\n[INFO] Resuming from incomplete iteration {incomplete_iteration}")
            starting_iteration = incomplete_iteration
        elif last_completed >= 0:
            print(f"\n[INFO] All previous iterations completed")
            starting_iteration = last_completed + 1
        else:
            print(f"\n[INFO] Starting from iteration 0")
            starting_iteration = 0
        
    else:
        # Create new experiment
        exp_name, exp_dir = get_experiment_name()
        
        # Ask if user wants to skip Phase 1 (LLM label generation)
        print("\n" + "="*70)
        print("  Phase 1: Pseudo Label Generation")
        print("="*70)
        
        skip_phase1 = False
        existing_labels_path = Path(base_config['llm']['cache_file'])
        
        if existing_labels_path.exists():
            print(f"\n[INFO] Found existing pseudo labels at: {existing_labels_path}")
            choice = input("\nOptions:\n  1. Reuse existing labels (skip Phase 1)\n  2. Regenerate labels\n\nYour choice (1/2): ").strip()
            
            if choice == '1':
                skip_phase1 = True
                print("\n[INFO] Will reuse existing pseudo labels")
                
                # Copy existing labels to experiment directory
                exp_labels_path = exp_dir / 'pseudo_labels.json'
                exp_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(existing_labels_path, exp_labels_path)
                print(f"[INFO] Copied labels to: {exp_labels_path}")
            elif choice == '2':
                print("\n[INFO] Will regenerate pseudo labels")
            else:
                print("[ERROR] Invalid choice")
                sys.exit(1)
        else:
            print("\n[INFO] No existing pseudo labels found. Will generate new labels.")
        
        # Create experiment-specific config
        exp_config, exp_config_path = create_experiment_config(base_config, exp_dir, iteration=0)
    
    # Check .env file
    if not Path('.env').exists() and not skip_phase1:
        print("\n[WARN] .env file not found")
        print("Please create .env file with your DASHSCOPE_API_KEY")
        print("Example: DASHSCOPE_API_KEY=your_key_here")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Confirm start
    print("\n" + "="*70)
    print(f"  Ready to start: {exp_name}")
    print("="*70)
    confirm = input("\nStart pipeline? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    # Determine starting iteration
    if mode == 'continue' and 'starting_iteration' in locals():
        iteration = starting_iteration
    else:
        iteration = 0
        starting_iteration = 0
    
    # Run iterations
    if iteration == 0:
        # Run initial training (iteration 0)
        cl_completed = run_training_iteration(
            exp_config, exp_config_path, 
            skip_phase1=skip_phase1, 
            skip_phase4=False,
            iteration=0
        )
        iteration = 1
    else:
        # Skip to the target iteration
        # Check if CL was completed in iteration 0
        cl_completed = (exp_dir / 'confident_learning' / 'pseudo_labels_cleaned.json').exists()
        if not cl_completed:
            print("\n[WARN] Confident Learning not completed. Cannot proceed to iteration 1+")
            print("[INFO] Skipping to evaluation...")
            iteration = starting_iteration + 1  # Exit loop
    
    # If Confident Learning was run, continue with iterations
    while cl_completed and iteration <= 5:
        # If continuing and we haven't passed the resumption point, run directly
        if mode == 'continue' and iteration <= starting_iteration:
            print(f"\n[INFO] Resuming iteration {iteration} with cleaned labels...")
            
            # Load or create config for this iteration
            iter_config_path = exp_dir / f'config_iter{iteration}.yaml'
            if iter_config_path.exists():
                with open(iter_config_path, 'r') as f:
                    exp_config = yaml.safe_load(f)
                exp_config_path = iter_config_path
            else:
                exp_config, exp_config_path = create_experiment_config(base_config, exp_dir, iteration=iteration)
            
            # Run training with cleaned labels
            cl_completed = run_training_iteration(
                exp_config, exp_config_path,
                skip_phase1=True,
                skip_phase4=True,
                iteration=iteration
            )
            
            iteration += 1
            
        # For new experiments or after resuming, ask user
        elif iteration > starting_iteration or mode != 'continue':
            print("\n" + "="*70)
            print("  Confident Learning Iteration Option")
            print("="*70)
            print("\nYou can now retrain with cleaned labels for better performance.")
            print("This will run Phase 2 (MAE) and Phase 3 (WSL) with cleaned labels.")
            print("Phase 4 (Confident Learning) will be skipped in this iteration.")
            
            choice = input("\nRetrain with cleaned labels? (y/n): ").strip().lower()
            if choice != 'y':
                print("\n[INFO] Skipping retraining. Pipeline completed.")
                break
            
            print(f"\n[INFO] Starting iteration {iteration} with cleaned labels...")
            
            # Create new config for this iteration
            exp_config, exp_config_path = create_experiment_config(base_config, exp_dir, iteration=iteration)
            
            # Run training with cleaned labels (skip Phase 1 and Phase 4)
            cl_completed = run_training_iteration(
                exp_config, exp_config_path,
                skip_phase1=True,  # Always skip Phase 1 in iterations
                skip_phase4=True,   # Always skip Phase 4 in iterations
                iteration=iteration
            )
            
            iteration += 1
        else:
            break
        
        # Safety check: limit iterations
        if iteration > 5:
            print("\n[WARN] Maximum iterations (5) reached. Stopping.")
            break
    
    # Run Evaluation
    print("\n" + "="*70)
    print("  Evaluation")
    print("="*70)
    
    # Evaluate all iterations
    available_configs = []
    if (exp_dir / 'config.yaml').exists():
        available_configs.append(('Initial Model', exp_dir / 'config.yaml'))
    
    for i in range(1, iteration):
        iter_config = exp_dir / f'config_iter{i}.yaml'
        if iter_config.exists():
            available_configs.append((f'Iteration {i} Model', iter_config))
    
    if available_configs:
        print("\nAvailable models for evaluation:")
        for i, (name, _) in enumerate(available_configs, 1):
            print(f"  {i}. {name}")
        print(f"  {len(available_configs)+1}. Evaluate all")
        print(f"  {len(available_configs)+2}. Skip evaluation")
        
        eval_choice = input(f"\nYour choice (1-{len(available_configs)+2}): ").strip()
        
        try:
            choice_idx = int(eval_choice)
            if 1 <= choice_idx <= len(available_configs):
                # Evaluate specific model
                name, config_file = available_configs[choice_idx - 1]
                success = run_command(
                    f"conda activate .\.conda && python evaluate.py --config {config_file}",
                    f"Evaluation: {name}"
                )
            elif choice_idx == len(available_configs) + 1:
                # Evaluate all
                for name, config_file in available_configs:
                    run_command(
                        f"conda activate .\.conda && python evaluate.py --config {config_file}",
                        f"Evaluation: {name}"
                    )
            else:
                print("\n[SKIP] Evaluation skipped")
        except ValueError:
            print("\n[SKIP] Invalid choice, evaluation skipped")
    
    # Summary
    print("\n" + "="*70)
    print("  [SUCCESS] Pipeline Completed Successfully!")
    print("="*70)
    print(f"\nExperiment: {exp_name}")
    print(f"Location: {exp_dir}")
    
    # Print output locations
    print("\n[OUTPUTS]")
    for i in range(iteration):
        suffix = "" if i == 0 else f"_iter{i}"
        config_name = "config.yaml" if i == 0 else f"config_iter{i}.yaml"
        print(f"\n  Iteration {i if i > 0 else 'Initial'}:")
        print(f"    - Config: {exp_dir / config_name}")
        print(f"    - MAE: {exp_dir / f'mae_pretrain{suffix}'}/encoder_best.pth")
        print(f"    - WSL: {exp_dir / f'wsl_train{suffix}'}/")
    
    if (exp_dir / 'confident_learning' / 'pseudo_labels_cleaned.json').exists():
        print(f"\n  Confident Learning:")
        print(f"    - Cleaned Labels: {exp_dir / 'confident_learning' / 'pseudo_labels_cleaned.json'}")
    
    print("\n[TENSORBOARD]")
    print(f"  Run: tensorboard --logdir={exp_dir}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARN] Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[ERROR] Pipeline failed with error:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
