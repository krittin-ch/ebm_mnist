import os
import torchvision.transforms.v2 as T
from ebm_modules import Backbone, ClassificationHead, ProjectionHead, LinearProbe
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from tools import *
import argparse
from tqdm import tqdm


def parse_config():
    parser = argparse.ArgumentParser(description="Training energy-based model")

    parser.add_argument("--shuffle", action="store_true", help="shuffle when called")
    parser.add_argument("--ssl", action="store_true", help="activate SSL mode")
    parser.add_argument(
        "--sup", action="store_true", help="activate supervised learning mode"
    )
    parser.add_argument("--train", action="store_true", help="train when called")
    parser.add_argument(
        "--linear_probe", action="store_true", help="linear probe training when called"
    )
    parser.add_argument("--test", action="store_true", help="test when called")
    parser.add_argument(
        "--scheduler", action="store_true", help="schedule learning rate"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")  # might need to change for macOS
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--weight", type=str, default=None, help="add model path")

    args = parser.parse_args()

    return args


def main():
    args = parse_config()

    device = "cuda:0" if not args.cpu and torch.cuda.is_available() else "cpu"

    mnist_transform = T.Compose(
        [
            T.RandomRotation(20),
            T.RandomAffine(
                degrees=15, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=10
            ),
            T.RandomPerspective(distortion_scale=0.3, p=0.5),
            T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ]
    )

    train_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=T.ToTensor()
    )

    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=T.ToTensor()
    )

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=args.shuffle
    )

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    if not (args.train or args.test or args.linear_probe):
        raise ValueError(
            "Please select `--train` for training, `--linear_probe` for linear probe training (set encoder weight), `--test` for testing"
        )

    # model setup
    act_layer = nn.ReLU

    if args.ssl:
        # train ssl
        encoder = Backbone(
            act_layer=act_layer
        )  # same as the online_model (for naming purpose)
        predictor = ProjectionHead(act_layer=act_layer)
        linear_probe = LinearProbe()

        target_model = Backbone(
            act_layer=act_layer
        )  # target model (get EMA update)
        target_model.load_state_dict(encoder.state_dict())

        if args.train:
            train_ssl(
                encoder=encoder,
                # predictor=predictor,
                target_model=target_model,
                train_loader=train_loader,
                optimizer_func=torch.optim.Adam,
                loss_func=encoder.contrastive_loss,
                device=device,
                transform=mnist_transform,
                epochs=args.epochs,
                save_path="weights_ssl",
                save_interval=args.save_interval,
                scheduler=args.scheduler,
            )

        # SSL -> Supervised
        if args.linear_probe:
            if not os.path.exists(args.weight):
                raise ValueError(
                    f"The path `{args.weight}` does not exist (no encoder availble)."
                )

            model_states = torch.load(args.weight)
            encoder.load_state_dict(model_states["encoder"])

            train_linear_probe(
                encoder=encoder,
                linear_probe=linear_probe,
                train_loader=train_loader,
                optimizer_func=torch.optim.Adam,
                loss_func=nn.CrossEntropyLoss(),
                device=device,
                epochs=args.epochs,
                save_path="weights_ssl_linear_probe",
                save_interval=args.save_interval,
                scheduler=args.scheduler,
            )

        if args.test:
            if not os.path.exists(args.weight):
                raise ValueError(
                    f"The path '{args.weight}' does not exist (no model state availble)."
                )

            model_states = torch.load(args.weight)
            encoder.load_state_dict(model_states["encoder"])
            linear_probe.load_state_dict(model_states["linear_probe"])

            test_linear_probe(
                encoder=encoder,
                linear_probe=linear_probe,
                test_loader=test_loader,
                device=device,
            )

    if args.sup:
        encoder = Backbone(act_layer=act_layer)
        classifier = ClassificationHead(act_layer=act_layer)

        encoder.to(device)
        classifier.to(device)

        if args.train:
            train_supervised(
                encoder=encoder,
                classifier=classifier,
                train_loader=train_loader,
                optimizer_func=torch.optim.Adam,
                loss_func=nn.CrossEntropyLoss(),
                device=device,
                epochs=args.epochs,
                save_path="weights_supervised",
                save_interval=args.save_interval,
                scheduler=args.scheduler,
            )

        if args.test:
            if args.train:
                print(
                    "The model has just been trained. The models are using these parameters."
                )
            else:
                if os.path.exists(args.weight):
                    model_states = torch.load(args.weight)
                    encoder.load_state_dict(model_states["encoder"])
                    classifier.load_state_dict(model_states["classifier"])
                else:
                    raise ValueError(
                        f"The path '{args.weight}' does not exist (no model state availble)."
                    )

            test_supervised(
                encoder=encoder,
                classifier=classifier,
                test_loader=test_loader,
                device=device,
            )

if __name__ == "__main__":
    main()
