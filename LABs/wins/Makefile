# Install exact Python and CUDA versions
conda-create:
	conda env create -f environment.yml

conda-update:
	conda env update --prune -f environment.yml
	# echo "!!!RUN RIGHT NOW:\nconda activate fsdl-text-recognizer-2021"

# Compile and install exact pip packages
pip-tools:
	pip install pip-tools
	pip-compile requirements/prod.in && pip-compile requirements/dev.in
	
	# install requirements for dev
	# python -m piptools sync requirements/prod.txt requirements/dev.txt
	# pip install -r requirements/prod.txt && pip install -r requirements/dev.txt
	# install requirements for prod
	# pip-sync

# Example training command
train-mnist-cnn-ddp:
	python training/run_experiment.py --max_epochs=10 --gpus=-1 --accelerator=ddp --num_workers=20 --data_class=MNIST --model_class=CNN

# Lint
lint:
	tasks/lint.sh
