from typing import Any, Dict, List, Literal, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DataSampler:
    def connect(self, trainer: pl.Trainer):
        self.trainer = trainer

    def setup(self, global_rank: int) -> None:
        raise NotImplementedError()

    def transform_dataloader(self, dataloader: DataLoader, consumed_samples: int = 0) -> DataLoader:
        raise NotImplementedError()


class MegatronDataSampler(DataSampler):
    def __init__(
        self,
        seq_len: int,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        dataloader_type: Literal["single", "cyclic"] = "single",
    ):
        self.seq_len = seq_len
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.rampup_batch_size = rampup_batch_size
        self.dataloader_type = dataloader_type
        self.init_consumed_samples: int = 0
        self.prev_consumed_samples = 0
        self.if_first_step = 0
        self.prev_global_batch_size = None

    def setup(self, global_rank: int) -> None:
        from nemo.lightning.data import setup_microbatch_calculator

        setup_microbatch_calculator(global_rank, self.micro_batch_size, self.global_batch_size, self.rampup_batch_size)

    def transform_dataloader(self, dataloader: DataLoader, consumed_samples: int = 0) -> DataLoader:
        from nemo.lightning.data import add_megatron_sampler

        return add_megatron_sampler(
            dataloader,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=self.rampup_batch_size,
            consumed_samples=consumed_samples,
            dataloader_type=self.dataloader_type,
        )

    def compute_consumed_samples(self, steps_since_resume=0) -> int:
        from nemo.lightning.pytorch.strategies import MegatronStrategy
        from nemo.utils import AppState

        if not isinstance(self.trainer.strategy, MegatronStrategy):
            return 0

        app_state = AppState()

        if self.rampup_batch_size is not None:
            from apex.transformer.pipeline_parallel.utils import _GLOBAL_NUM_MICROBATCHES_CALCULATOR

            current_global_batch_size = getattr(_GLOBAL_NUM_MICROBATCHES_CALCULATOR, "current_global_batch_size", 1)
            consumed_samples = self.prev_consumed_samples + self.if_first_step * current_global_batch_size
        else:
            consumed_samples = (
                self.init_consumed_samples
                + steps_since_resume * app_state.data_parallel_size * self.micro_batch_size * self.num_microbatches
            )

        return int(consumed_samples)

    # Megatron callbacks
    def on_megatron_step_start(self, trainer: pl.Trainer) -> None:
        # do validation and save the checkpoint when gbs is changed
        if (
            self.rampup_batch_size is not None
            and self.prev_global_batch_size != self.current_global_batch_size
            and self.prev_global_batch_size
        ):
            trainer.should_stop = True

    def on_megatron_step_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        import apex.transformer.pipeline_parallel.utils

        if self.rampup_batch_size is None:
            return

        self.prev_global_batch_size = self.current_global_batch_size

        # TODO: Add consumed samples
        consumed_samples = self.compute_consumed_samples(trainer.global_step + 1 - self.init_global_step)

        self.prev_consumed_samples = consumed_samples

        num_microbatch_calculator = (
            apex.transformer.pipeline_parallel.utils._GLOBAL_NUM_MICROBATCHES_CALCULATOR  # noqa: SLF001
        )

        num_microbatch_calculator.update(
            consumed_samples=consumed_samples, consistency_check=False,
        )
        current_global_batch_size = num_microbatch_calculator.current_global_batch_size
        pl_module.log(
            "global_batch_size", current_global_batch_size, prog_bar=True, rank_zero_only=True, batch_size=1,
        )
        self.if_first_step = 1

    @property
    def megatron_data_kwargs(self) -> Dict[str, Any]:
        return {
            "seq_length": self.seq_len,
            "micro_batch_size": self.micro_batch_size,
            "num_microbatches": self.num_microbatches,
        }

    @property
    def num_microbatches(self) -> int:
        from apex.transformer.pipeline_parallel.utils import get_num_microbatches

        return get_num_microbatches()

    @property
    def current_global_batch_size(self) -> int:
        import apex.transformer.pipeline_parallel.utils

        num_microbatch_calculator = (
            apex.transformer.pipeline_parallel.utils._GLOBAL_NUM_MICROBATCHES_CALCULATOR  # noqa: SLF001
        )
        current_global_batch_size = num_microbatch_calculator.current_global_batch_size

        return current_global_batch_size
