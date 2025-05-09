import click
import sys
# Add project root to sys.path if needed, e.g., if Membership-Inference is a subdir
import attack

@click.group()
def cli():
    pass

@cli.group()
def membership_inference():
    pass

# Command for running with pre-trained models (assumes models are already trained and saved)
@membership_inference.command(help='Membership Inference Attack with pre-trained target and shadow models.')
@click.option('--dataset', default='CIFAR10', type=click.Choice(['CIFAR10', 'MNIST']), help='Which dataset to use (CIFAR10 or MNIST)')
@click.option('--data-path', default='../data', type=click.Path(exists=True), help='Path to store/load data') # Adjusted default, ensure this path exists
@click.option('--model-path', default='../models_output', type=click.Path(), help='Path to save or load model checkpoints') # Adjusted default
@click.option('--topk', is_flag=True, help='Flag to enable using Top K posteriors for attack data.')
@click.option('--param-init', is_flag=True, help='Flag to enable custom model params initialization (for loading if needed, though typically for training).')
@click.option('--verbose', is_flag=True, help='Add Verbosity.')
@click.option('--no-early-stopping', is_flag=True, help='Disable early stopping during model training.')
def pretrained_dummy(dataset, data_path, model_path, topk, param_init, verbose, no_early_stopping):
    click.echo(f'Performing Membership Inference with Pre-trained Models on {dataset}')
    # When using pre-trained, trainTargetModel and trainShadowModel are False
    attack.create_attack(
        dataset_name=dataset,
        data_path_root=data_path,
        model_path_root=model_path,
        train_target_model_flag=False,
        train_shadow_model_flag=False,
        need_augm_flag=False, # Augmentation not applicable if only loading
        need_topk_posteriors=topk,
        param_init_flag=param_init,
        verbose_flag=verbose,
        use_early_stopping_flag=not no_early_stopping
    )

# Command for running with training enabled for target and shadow models
@membership_inference.command(help='Membership Inference Attack with training enabled for target and shadow models.')
@click.option('--dataset', default='CIFAR10', type=click.Choice(['CIFAR10', 'MNIST']), help='Which dataset to use (CIFAR10 or MNIST)')
@click.option('--data-path', default='../data', type=click.Path(exists=True), help='Path to store/load data')
@click.option('--model-path', default='../models_output', type=click.Path(), help='Path to save or load model checkpoints')
@click.option('--no-train-target', is_flag=True, help='Do not train the target model (load if exists).')
@click.option('--no-train-shadow', is_flag=True, help='Do not train the shadow model (load if exists).')
@click.option('--augm', is_flag=True, help='To use data augmentation on target and shadow training set or not.')
@click.option('--topk', is_flag=True, help='Flag to enable using Top K posteriors for attack data.')
@click.option('--param-init', is_flag=True, help='Flag to enable custom model params initialization.')
@click.option('--verbose', is_flag=True, help='Add Verbosity.')
@click.option('--no-early-stopping', is_flag=True, help='Disable early stopping during model training.')
def train_dummy(dataset, data_path, model_path, no_train_target, no_train_shadow, augm, topk, param_init, verbose, no_early_stopping):
    click.echo(f'Performing Membership Inference with Training Enabled on {dataset}')
    attack.create_attack(
        dataset_name=dataset,
        data_path_root=data_path,
        model_path_root=model_path,
        train_target_model_flag=not no_train_target,
        train_shadow_model_flag=not no_train_shadow,
        need_augm_flag=augm,
        need_topk_posteriors=topk,
        param_init_flag=param_init,
        verbose_flag=verbose,
        use_early_stopping_flag=not no_early_stopping
    )


if __name__ == '__main__':
    cli()