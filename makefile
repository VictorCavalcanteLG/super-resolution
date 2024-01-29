include .env
export $(shell sed 's/=.*//' .env)

train_decoder_model:
	$ wandb docker-run --name train_decoder_model --env MODEL_CONFIG_PATH=$(MODEL_CONFIG_PATH) --gpus all -it --rm -v .:/super-resolution/ pytorch-gpu

train_local_decoder_model:
	$ python -m src.scripts.train_model

# generate_images:
#     $ wandb docker-run --name generate_images --env MODEL_CONFIG_PATH=$(MODEL_CONFIG_PATH) --gpus all -it --
