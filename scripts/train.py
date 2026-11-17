import faulthandler
import os
import signal

faulthandler.register(signal.SIGUSR1)
if os.environ.get('IMAGEWAM_DEBUG_STARTUP'):
    faulthandler.dump_traceback_later(180, repeat=True)

import hydra
from omegaconf import DictConfig

from imagewam.runtime import run_training
from imagewam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
