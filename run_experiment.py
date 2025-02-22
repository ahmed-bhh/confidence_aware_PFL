# import json
# import click
# from vem import train
#
#
# @click.command()
# @click.option(
#     "--dataset",
#     default="CIFAR10",
#     type=click.Choice(["CIFAR10", "CIFAR100"], case_sensitive=False),
# )
# @click.option(
#     "--model",
#     default="CNNCifar",
#     type=click.Choice(["CNNCifar", "CNNCifar100"], case_sensitive=False),
# )
# @click.option("--batch_size", default=10000, help="Batch size.")
# @click.option("--lr_head", default=0.0003, help="Lr for base model.")
# @click.option("--lr_base", default=0.0001, help="Lr for local model.")
# @click.option("--momentum", default=0.9, help="Momentum for optimizer.")
# @click.option("--head_epochs", default=10, help="Number of epochs for head network training.")
# @click.option("--base_epochs", default=10, help="Number of epochs for base network training.")
# @click.option("--n_mc", default=5, help="number of classes in each client.")
# @click.option("--scale", default=1, help="Initialized scale of Gaussian posterior.")
# @click.option("--beta", default=0, help="Momentum for update of the global model.")
# @click.option("--n_labels", default=2, help="Number of classes in each local dataset.")
# @click.option("--relabel", is_flag=True)
# @click.option("--n_rounds", default=100, help="Number of communication rounds.")
# @click.option("--max_data", default=0, help="The number of data points for the overall dataset.")
# @click.option("--n_clients", default=500, help="Number of clients.")
# @click.option("--sampling_rate", default=0.1, help="Clients sampling rate.")
# @click.option("--path_to_data", default="./data")
# @click.option("--seed", default=0, help="Random seed.")
# @click.option(
#     "--config", help="Path to the configuration file.", default=None,
# )
# def main(**kwargs):
#     if kwargs["config"]:
#         with open(kwargs["config"]) as f:
#             kwargs = json.load(f)
#     else:
#         del kwargs["config"]
#
#     print(kwargs)
#     train(**kwargs)
#
#
# if __name__ == "__main__":
#     main()
#
# import yaml
# import click
# import json
# from vem import train
#
# @click.command()
# @click.option(
#     "--dataset",
#     default="CIFAR10",
#     type=click.Choice(["CIFAR10", "CIFAR100"], case_sensitive=False),
# )
# @click.option(
#     "--model",
#     default="CNNCifar",
#     type=click.Choice(["CNNCifar", "CNNCifar100"], case_sensitive=False),
# )
# @click.option("--batch_size", default=10000, help="Batch size.")
# @click.option("--lr_head", default=0.0003, help="Lr for base model.")
# @click.option("--lr_base", default=0.0001, help="Lr for local model.")
# @click.option("--momentum", default=0.9, help="Momentum for optimizer.")
# @click.option("--head_epochs", default=10, help="Number of epochs for head network training.")
# @click.option("--base_epochs", default=10, help="Number of epochs for base network training.")
# @click.option("--n_mc", default=5, help="Number of Monte Carlo samples.")
# @click.option("--scale", default=1, help="Initialized scale of Gaussian posterior.")
# @click.option("--beta", default=0, help="Momentum for update of the global model.")
# @click.option("--n_labels", default=2, help="Number of classes in each local dataset.")
# @click.option("--relabel", is_flag=True)
# @click.option("--n_rounds", default=100, help="Number of communication rounds.")
# @click.option("--max_data", default=0, help="The number of data points for the overall dataset.")
# @click.option("--n_clients", default=500, help="Number of clients.")
# @click.option("--sampling_rate", default=0.1, help="Clients sampling rate.")
# @click.option("--path_to_data", default="./data")
# @click.option("--seed", default=0, help="Random seed.")
# @click.option(
#     "--config", help="Path to the configuration file.", default=None,
# )
# def main(**kwargs):
#     if kwargs.get("config"):
#         # Charger le fichier YAML au lieu de JSON
#         with open(kwargs["config"], "r") as f:
#             yaml_content = yaml.safe_load(f)  # Charger le contenu YAML
#             kwargs.update(yaml_content)  # Ajouter les clés du fichier YAML aux arguments
#         del kwargs["config"]  # Supprimer l'argument 'config'
#
#     print(kwargs)
#     train("**kwargs")
#
# if __name__ == "__main__":
#     main()

#
# import json
# import yaml  # Importer PyYAML
# import click
# from vem import train
#
#
# @click.command()
# @click.option(
#     "--config", help="Path to the configuration file (JSON or YAML).", default=None,
# )
# def main(**kwargs):
#     if kwargs["config"]:
#         with open(kwargs["config"], 'r') as f:
#             # Détecter l'extension du fichier pour choisir le parseur
#             if kwargs["config"].endswith(('.yaml', '.yml')):
#                 config_data = yaml.safe_load(f)
#             else:
#                 config_data = json.load(f)
#         kwargs.update(config_data)
#     # Supprimer la clé 'config' après chargement
#     kwargs.pop("config", None)
#
#     print(kwargs)
#     train(**kwargs)
#
#
# if __name__ == "__main__":
#     main()
import json
import yaml  # Importer PyYAML
import click
import os
import sys
sys.path.append("../vem")
print(os.getcwd())
from vem.pfedvem import train

def parse_yaml_to_flat_dict(config_data):
    """
    Convertit un fichier YAML structuré en un dictionnaire aplati avec des clés spécifiques.
    """
    flat_config = {
        "model": config_data["model_params"]["model_name"],
        "dataset": config_data["data_params"]["dataset_name"],
        "path_to_data": config_data["data_params"]["root_path"],
        "batch_size": config_data["data_params"]["train_batch_size"],
        "lr_head": config_data["optimization"]["learning_rate"],
        "lr_base": config_data["optimization"]["personal_learning_rate"],
        "base_epochs": config_data["optimization"]["local_epochs"],
        "momentum": 0.9,  # Momentum est manquant dans le YAML, on peut l'ajouter comme valeur par défaut
        "n_labels": config_data["data_params"]["specific_dataset_params"]["classes_per_user"],
        "n_rounds": config_data["optimization"]["global_iters"],
        "n_clients": config_data["data_params"]["specific_dataset_params"]["n_clients"],
        "sampling_rate": config_data["train_params"]["num_clients_per_round"] / config_data["data_params"]["specific_dataset_params"]["n_clients"],
        "seed": config_data["train_params"]["seeds"][0],
        "relabel": False,  # Relabel est manquant dans le YAML, on peut l'ajouter comme valeur par défaut
        "head_epochs": 10,  # Valeur par défaut ajoutée
        "scale": config_data["model_params"]["weight_scale"],
        "max_data": config_data["data_params"]["max_dataset_size_per_user"],
        "beta": config_data["model_params"]["beta"],
        "n_mc": 5,  # Valeur par défaut ajoutée
    }
    return flat_config


@click.command()
@click.option(
    "--config", help="Path to the configuration file (YAML).", default=None,
)
def main(**kwargs):
    if kwargs["config"]:
        with open(kwargs["config"], 'r') as f:
            # Charger le YAML
            config_data = yaml.safe_load(f)
            # Transformer en dictionnaire aplati
            flat_config = parse_yaml_to_flat_dict(config_data)
            kwargs.update(flat_config)
    # Supprimer la clé 'config' après chargement
    kwargs.pop("config", None)

    print(kwargs)  # Affichage pour vérification
    train(**kwargs)


if __name__ == "__main__":
    main()
