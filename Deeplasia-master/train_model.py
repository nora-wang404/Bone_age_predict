import logging, logging.config
from lib.utils.log import LOG_CONFIG

logging.config.dictConfig(LOG_CONFIG)

from lib.utils.cli import CustomCli
import sys

sys.path.append("..")

from lib import testing
from lib.models import *
from lib.datasets import *
#python train_model.py --config=configs/defaults.yml --trainer.max_epochs=100 --model.backbone=efficientnet-b4 --data.annotation_path=/root/autodl-tmp/guling/boneage-training-dataset.csv --data.img_dir=/root/autodl-tmp/guling --data.num_workers=8 --data.train_batch_size=30 --data.test_batch_size=30

# --config=configs/defaults.yml
# --trainer.max_epochs=100
# --model.backbone=efficientnet-b4
# --data.annotation_path=/media/dzy/deep1/train_data2/guling/boneage-training-dataset.csv
# --data.img_dir=/media/dzy/deep1/train_data2/guling
# --data.num_workers=8
# --data.train_batch_size=16
# --data.test_batch_size=30
def main():
    logger = logging.getLogger()
    cli = CustomCli(
        BoneAgeModel,
        HandDatamodule,
        run=False,
        parser_kwargs={"default_config_files": ["configs/defaults.yml"],},
    )
    cli.setup_callbacks()
    cli.log_info()
    try:
        cli.examples_to_tb()
        logger.info(f"{'=' * 10} start training {'=' * 10}")
        cli.trainer.fit(cli.model, cli.datamodule)
        cli.log_train_stats()

        logger.info(f"{'=' * 10} Testing model {'=' * 10}")
        test_ckp_path = cli.get_model_weights()
    except Exception:
        logger.exception("No training samples, testing only")
        test_ckp_path = cli.config["trainer"]["resume_from_checkpoint"]

    log_dict = testing.evaluate_bone_age_model(
        test_ckp_path, cli.config, cli.trainer.logger.log_dir, cli.trainer
    )
    cli.model.logger.log_metrics(log_dict)
    cli.model.logger.save()
    logger.info(f"======= END =========")


if __name__ == "__main__":
    main()
