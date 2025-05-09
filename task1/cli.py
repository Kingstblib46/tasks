import click
import sys
# Add project root to sys.path if needed, e.g., if Membership-Inference is a subdir
# sys.path.insert(1, 'Membership-Inference') # [cite: 1] # Or adjust based on your project structure
import attack # [cite: 1]

@click.group()
def cli(): # [cite: 1]
    pass

@cli.group()
def membership_inference(): # [cite: 1]
    pass

# Command for running with pre-trained models (assumes models are already trained and saved)
@membership_inference.command(help='Membership Inference Attack with pre-trained target and shadow models.') # [cite: 1]
@click.option('--dataset', default='CIFAR10', type=click.Choice(['CIFAR10', 'MNIST']), help='Which dataset to use (CIFAR10 or MNIST)') # [cite: 1]
@click.option('--data-path', default='../data', type=click.Path(exists=True), help='Path to store/load data') # [cite: 1] # Adjusted default, ensure this path exists
@click.option('--model-path', default='../models_output', type=click.Path(), help='Path to save or load model checkpoints') # [cite: 1] # Adjusted default
@click.option('--topk', is_flag=True, help='Flag to enable using Top K posteriors for attack data.')
@click.option('--param-init', is_flag=True, help='Flag to enable custom model params initialization (for loading if needed, though typically for training).')
@click.option('--verbose', is_flag=True, help='Add Verbosity.')
@click.option('--no-early-stopping', is_flag=True, help='Disable early stopping during model training.')
def pretrained_dummy(dataset, data_path, model_path, topk, param_init, verbose, no_early_stopping):
    click.echo(f'Performing Membership Inference with Pre-trained Models on {dataset}') # [cite: 2]
    # When using pre-trained, trainTargetModel and trainShadowModel are False
    attack.create_attack(
        dataset_name=dataset,
        data_path_root=data_path,
        model_path_root=model_path,
        train_target_model_flag=False, # [cite: 2]
        train_shadow_model_flag=False, # [cite: 2]
        need_augm_flag=False, # Augmentation not applicable if only loading [cite: 2]
        need_topk_posteriors=topk, # [cite: 2]
        param_init_flag=param_init, # [cite: 2]
        verbose_flag=verbose, # [cite: 2]
        use_early_stopping_flag=not no_early_stopping
    )

# Command for running with training enabled for target and shadow models
@membership_inference.command(help='Membership Inference Attack with training enabled for target and shadow models.') # [cite: 3]
@click.option('--dataset', default='CIFAR10', type=click.Choice(['CIFAR10', 'MNIST']), help='Which dataset to use (CIFAR10 or MNIST)') # [cite: 3]
@click.option('--data-path', default='../data', type=click.Path(exists=True), help='Path to store/load data') # [cite: 3]
@click.option('--model-path', default='../models_output', type=click.Path(), help='Path to save or load model checkpoints') # [cite: 3]
@click.option('--no-train-target', is_flag=True, help='Do not train the target model (load if exists).')
@click.option('--no-train-shadow', is_flag=True, help='Do not train the shadow model (load if exists).')
@click.option('--augm', is_flag=True, help='To use data augmentation on target and shadow training set or not.') # [cite: 5]
@click.option('--topk', is_flag=True, help='Flag to enable using Top K posteriors for attack data.') # [cite: 5]
@click.option('--param-init', is_flag=True, help='Flag to enable custom model params initialization.') # [cite: 5]
@click.option('--verbose', is_flag=True, help='Add Verbosity.') # [cite: 5]
@click.option('--no-early-stopping', is_flag=True, help='Disable early stopping during model training.')
def train_dummy(dataset, data_path, model_path, no_train_target, no_train_shadow, augm, topk, param_init, verbose, no_early_stopping):
    click.echo(f'Performing Membership Inference with Training Enabled on {dataset}') # [cite: 4]
    attack.create_attack(
        dataset_name=dataset,
        data_path_root=data_path,
        model_path_root=model_path,
        train_target_model_flag=not no_train_target, # [cite: 4, 6]
        train_shadow_model_flag=not no_train_shadow, # [cite: 4, 6]
        need_augm_flag=augm, # [cite: 6]
        need_topk_posteriors=topk, # [cite: 6]
        param_init_flag=param_init, # [cite: 6]
        verbose_flag=verbose, # [cite: 6]
        use_early_stopping_flag=not no_early_stopping
    )


if __name__ == '__main__':
    cli() # [cite: 6]