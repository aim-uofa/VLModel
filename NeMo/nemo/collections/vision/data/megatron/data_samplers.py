# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, List

import torch
from torch.utils.data import Dataset

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingRandomSampler
from nemo.collections.vision.data.megatron.vit_dataset import RandomSeedDataset
import math

class MegatronVisionPretrainingRandomSampler(MegatronPretrainingRandomSampler):
    def __init__(
        self,
        dataset: Dataset,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        data_sharding: bool,
        drop_last: bool = True,
        global_batch_size: Optional[int] = None,
        pad_samples_to_global_batch_size: Optional[bool] = False,
    ) -> None:
        super().__init__(
            total_samples=total_samples,
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            drop_last=drop_last,
            global_batch_size=global_batch_size,
            pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
        )
        self.dataset = dataset
        self.data_sharding = data_sharding

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = (
                self.total_samples // self.micro_batch_times_data_parallel_size
            ) * self.micro_batch_times_data_parallel_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.data_parallel_rank :: self.data_parallel_size]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            yield batch

def xtuner_get_length_grouped_indices(lengths, group_batch_size, generator=None):

    def process(lengths, group_batch_size, generator=None):
        indices = torch.randperm(len(lengths), generator=generator)
        megabatches = [
            indices[i:i + group_batch_size].tolist()
            for i in range(0, len(lengths), group_batch_size)
        ]
        megabatches = [
            sorted(megabatch, key=lambda i: lengths[i], reverse=True)
            for megabatch in megabatches
        ]
        return megabatches

    assert all(leng != 0 for leng in lengths), 'Should not have zero length.'
    if all(leng > 0 for leng in lengths) or all(leng < 0 for leng in lengths):
        # all samples are in the same modality
        megabatches = process(lengths, group_batch_size, generator=generator)
    else:
        mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths)
                                       if l > 0])
        lang_indices, lang_lengths = zip(*[(i, -l)
                                           for i, l in enumerate(lengths)
                                           if l < 0])
        mm_megabatches = []
        for mm_megabatch in process(
                mm_lengths, group_batch_size, generator=generator):
            mm_megabatches.append([mm_indices[i] for i in mm_megabatch])
        lang_megabatches = []
        for lang_megabatch in process(
                lang_lengths, group_batch_size, generator=generator):
            lang_megabatches.append([lang_indices[i] for i in lang_megabatch])

        last_mm = mm_megabatches[-1]
        last_lang = lang_megabatches[-1]
        last_batch = last_mm + last_lang
        megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]

        megabatch_indices = torch.randperm(
            len(megabatches), generator=generator)
        megabatches = [megabatches[i] for i in megabatch_indices]

        if len(last_batch) > 0:
            megabatches.append(
                sorted(
                    last_batch, key=lambda i: abs(lengths[i]), reverse=True))

    # The rest is to get the biggest batch first.
    # Since each megabatch is sorted by descending length,
    # the longest element is the first
    megabatch_maximums = [
        abs(lengths[megabatch[0]]) for megabatch in megabatches
    ]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    # Switch to put the longest element in first position
    megabatches[0][0], megabatches[max_idx][0] = megabatches[max_idx][
        0], megabatches[0][0]

    return [i for megabatch in megabatches for i in megabatch]

class MegatronModalityPretrainingRandomSampler(MegatronPretrainingRandomSampler):
    def __init__(
        self,
        dataset: Dataset,
        total_samples: int,
        consumed_samples: int,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        data_sharding: bool,
        drop_last: bool = True,
        mega_batch_mult: Optional[int] = None,
        global_batch_size: Optional[int] = None,
        lengths: Optional[List[int]] = None,
        pad_samples_to_global_batch_size: Optional[bool] = False,
    ) -> None:
        super().__init__(
            total_samples=total_samples,
            consumed_samples=consumed_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            drop_last=drop_last,
            global_batch_size=global_batch_size,
            pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
        )
        self.dataset = dataset
        self.lengths = lengths
        self.data_sharding = data_sharding
        # calculate the last_batch_size when using/not using gradient accumulation
        if self.global_batch_size is not None:
            self.last_batch_size = self.total_samples % self.global_batch_size
        
        if self.global_batch_size is None:
            total_batch_size = self.micro_batch_size * self.data_parallel_size
        else:
            total_batch_size = self.global_batch_size
        
        if mega_batch_mult is None:
            # Default for mega_batch_mult: 50 or the number to get 4
            # megabatches, whichever is smaller.
            mega_batch_mult = min(
                len(self.dataset) // (total_batch_size * 4), 50)
            # Just in case, for tiny datasets
            if mega_batch_mult == 0:
                mega_batch_mult = 1
        self.group_batch_size = mega_batch_mult * total_batch_size
        self.num_samples = math.ceil(len(self.dataset) / self.data_parallel_size)

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        full_bucket_size = (self.total_samples // self.micro_batch_size) * self.micro_batch_size
        full_bucket_offset = current_epoch_samples
        g = torch.Generator()
        g.manual_seed(self.epoch)
        
        indices = xtuner_get_length_grouped_indices(
            lengths=self.lengths,
            group_batch_size=self.group_batch_size,
            generator=g)
        assert len(set(indices)) == len(indices)
        
        idx_range_active = indices[full_bucket_offset:]
        idx_range = idx_range_active[self.data_parallel_rank :: self.data_parallel_size]
        
        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            yield batch