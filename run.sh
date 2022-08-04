#!/usr/bin/env bash

python main.py --config DigitFive.yaml --base-path /media/document/ --communication_rounds 1 --dataset DigitFive --target-domain svhn --forget_rate 0.08
python main.py --config DigitFive.yaml --base-path /media/document/ --communication_rounds 1 --dataset DigitFive --target-domain mnistm --forget_rate 0.4
python main.py --config DigitFive.yaml --base-path /media/document/ --communication_rounds 1 --dataset DigitFive --target-domain syn --forget_rate 0.24
python main.py --config DigitFive.yaml --base-path /media/document/ --communication_rounds 1 --dataset DigitFive --target-domain usps --forget_rate 0.04
python main.py --config DigitFive.yaml --base-path /media/document/ --communication_rounds 1 --dataset DigitFive --target-domain mnist --forget_rate 0.04
