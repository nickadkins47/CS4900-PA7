
# use "python", "uv run", etc
PY = uv run

.PHONY: clean

model_names = PPO DQN DDPG SAC
train_all: $(addprefix models/, $(addsuffix .zip, $(model_names)))

train_% models/%.zip: | models
	$(PY) src/train.py $*

run_%: models/%.zip
	$(PY) src/run.py $*

clean:
	@rm -rf models

models:
	@mkdir -p $@