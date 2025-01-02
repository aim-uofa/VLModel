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

import os
from collections import OrderedDict
from statistics import mode
from typing import Dict, Optional

import torch

from nemo.collections.asr.parts.utils.offline_clustering import get_argmin_mat
from nemo.collections.asr.parts.utils.speaker_utils import convert_rttm_line, prepare_split_data
from nemo.collections.common.parts.preprocessing.collections import DiarizationSpeechLabel
from nemo.core.classes import Dataset
from nemo.core.neural_types import AudioSignal, EncodedRepresentation, LengthsType, NeuralType, ProbsType


def get_scale_mapping_list(uniq_timestamps):
    """
    Call get_argmin_mat function to find the index of the non-base-scale segment that is closest to the
    given base-scale segment. For each scale and each segment, a base-scale segment is assigned.

    Args:
        uniq_timestamps: (dict)
            The dictionary containing embeddings, timestamps and multiscale weights.
            If uniq_timestamps contains only one scale, single scale diarization is performed.

    Returns:
        scale_mapping_argmat (torch.tensor):

            The element at the m-th row and the n-th column of the scale mapping matrix indicates the (m+1)-th scale
            segment index which has the closest center distance with (n+1)-th segment in the base scale.

            - Example:
                `scale_mapping_argmat[2][101] = 85`

            In the above example, the code snippet means that 86-th segment in the 3rd scale (python index is 2) is
            mapped to the 102-th segment in the base scale. Thus, the longer segments bound to have more repeating
            numbers since multiple base scale segments (since the base scale has the shortest length) fall into the
            range of the longer segments. At the same time, each row contains N numbers of indices where N is number
            of segments in the base-scale (i.e., the finest scale).
    """
    timestamps_in_scales = []
    for key, val in uniq_timestamps['scale_dict'].items():
        timestamps_in_scales.append(torch.tensor(val['time_stamps']))
    session_scale_mapping_list = get_argmin_mat(timestamps_in_scales)
    scale_mapping_argmat = [[] for _ in range(len(uniq_timestamps['scale_dict'].keys()))]
    for scale_idx in range(len(session_scale_mapping_list)):
        scale_mapping_argmat[scale_idx] = session_scale_mapping_list[scale_idx]
    scale_mapping_argmat = torch.stack(scale_mapping_argmat)
    return scale_mapping_argmat


def extract_seg_info_from_rttm(uniq_id, rttm_lines, mapping_dict=None, target_spks=None):
    """
    Get RTTM lines containing speaker labels, start time and end time. target_spks contains two targeted
    speaker indices for creating groundtruth label files. Only speakers in target_spks variable will be
    included in the output lists.

    Args:
        uniq_id (str):
            Unique file ID that refers to an input audio file and corresponding RTTM (Annotation) file.
        rttm_lines (list):
            List containing RTTM lines in str format.
        mapping_dict (dict):
            Mapping between the estimated speakers and the speakers in the ground-truth annotation.
            `mapping_dict` variable is only provided when the inference mode is running in sequence-eval mode.
            Sequence eval mode uses the mapping between the estimated speakers and the speakers in ground-truth annotation.
    Returns:
        rttm_tup (tuple):
            Tuple containing lists of start time, end time and speaker labels.

    """
    stt_list, end_list, speaker_list, pairwise_infer_spks = [], [], [], []
    if target_spks:
        inv_map = {v: k for k, v in mapping_dict.items()}
        for spk_idx in target_spks:
            spk_str = f'speaker_{spk_idx}'
            if spk_str in inv_map:
                pairwise_infer_spks.append(inv_map[spk_str])
    for rttm_line in rttm_lines:
        start, end, speaker = convert_rttm_line(rttm_line)
        if target_spks is None or speaker in pairwise_infer_spks:
            end_list.append(end)
            stt_list.append(start)
            speaker_list.append(speaker)
    rttm_tup = (stt_list, end_list, speaker_list)
    return rttm_tup


def assign_frame_level_spk_vector(rttm_timestamps, round_digits, frame_per_sec, target_spks, min_spks=2):
    """
    Create a multi-dimensional vector sequence containing speaker timestamp information in RTTM.
    The unit-length is the frame shift length of the acoustic feature. The feature-level annotations
    `fr_level_target` will later be converted to base-segment level diarization label.

    Args:
        rttm_timestamps (list):
            List containing start and end time for each speaker segment label.
            stt_list, end_list and speaker_list are contained.
        frame_per_sec (int):
            Number of feature frames per second. This quantity is determined by window_stride variable in preprocessing module.
        target_spks (tuple):
            Speaker indices that are generated from combinations. If there are only one or two speakers,
            only a single target_spks variable is generated.

    Returns:
        fr_level_target (torch.tensor):
            Tensor containing label for each feature level frame.
    """
    stt_list, end_list, speaker_list = rttm_timestamps
    if len(speaker_list) == 0:
        return None
    else:
        sorted_speakers = sorted(list(set(speaker_list)))
        total_fr_len = int(max(end_list) * (10 ** round_digits))
        spk_num = max(len(sorted_speakers), min_spks)
        speaker_mapping_dict = {rttm_key: x_int for x_int, rttm_key in enumerate(sorted_speakers)}
        fr_level_target = torch.zeros(total_fr_len, spk_num)

        # If RTTM is not provided, then there is no speaker mapping dict in target_spks.
        # Thus, return a zero-filled tensor as a placeholder.
        for count, (stt, end, spk_rttm_key) in enumerate(zip(stt_list, end_list, speaker_list)):
            stt, end = round(stt, round_digits), round(end, round_digits)
            spk = speaker_mapping_dict[spk_rttm_key]
            stt_fr, end_fr = int(round(stt, 2) * frame_per_sec), int(round(end, round_digits) * frame_per_sec)
            fr_level_target[stt_fr:end_fr, spk] = 1
        return fr_level_target


class _AudioMSDDTrainDataset(Dataset):
    """
    Dataset class that loads a json file containing paths to audio files,
    RTTM files and number of speakers. This Dataset class is designed for
    training or fine-tuning speaker embedding extractor and diarization decoder
    at the same time.

    Example:
    {"audio_filepath": "/path/to/audio_0.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_0.rttm}
    ...
    {"audio_filepath": "/path/to/audio_n.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_n.rttm}

    Args:
        manifest_filepath (str):
            Path to input manifest json files.
        multiscale_args_dict (dict):
            Dictionary containing the parameters for multiscale segmentation and clustering.
        emb_dir (str):
            Path to a temporary folder where segmentation information for embedding extraction is saved.
        soft_label_thres (float):
            Threshold that determines the label of each segment based on RTTM file information.
        featurizer:
            Featurizer instance for generating features from the raw waveform.
        window_stride (float):
            Window stride for acoustic feature. This value is used for calculating the numbers of feature-level frames.
        emb_batch_size (int):
            Number of embedding vectors that are trained with attached computational graphs.
        pairwise_infer (bool):
            This variable should be True if dataloader is created for an inference task.
        random_flip (bool):
            If True, the two labels and input signals are randomly flipped per every epoch while training.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        output_types = {
            "features": NeuralType(('B', 'T'), AudioSignal()),
            "feature_length": NeuralType(('B'), LengthsType()),
            "ms_seg_timestamps": NeuralType(('B', 'C', 'T', 'D'), LengthsType()),
            "ms_seg_counts": NeuralType(('B', 'C'), LengthsType()),
            "clus_label_index": NeuralType(('B', 'T'), LengthsType()),
            "scale_mapping": NeuralType(('B', 'C', 'T'), LengthsType()),
            "targets": NeuralType(('B', 'T', 'C'), ProbsType()),
        }

        return output_types

    def __init__(
        self,
        *,
        manifest_filepath: str,
        multiscale_args_dict: str,
        emb_dir: str,
        soft_label_thres: float,
        featurizer,
        window_stride,
        emb_batch_size,
        pairwise_infer: bool,
        random_flip: bool = True,
        global_rank: int = 0,
    ):
        super().__init__()
        self.collection = DiarizationSpeechLabel(
            manifests_files=manifest_filepath.split(','),
            emb_dict=None,
            clus_label_dict=None,
            pairwise_infer=pairwise_infer,
        )
        self.featurizer = featurizer
        self.multiscale_args_dict = multiscale_args_dict
        self.emb_dir = emb_dir
        self.round_digits = 2
        self.decim = 10 ** self.round_digits
        self.soft_label_thres = soft_label_thres
        self.pairwise_infer = pairwise_infer
        self.max_spks = 2
        self.frame_per_sec = int(1 / window_stride)
        self.emb_batch_size = emb_batch_size
        self.random_flip = random_flip
        self.global_rank = global_rank
        self.manifest_filepath = manifest_filepath
        self.multiscale_timestamp_dict = prepare_split_data(
            self.manifest_filepath, self.emb_dir, self.multiscale_args_dict, self.global_rank,
        )

    def __len__(self):
        return len(self.collection)

    def assign_labels_to_longer_segs(self, uniq_id, base_scale_clus_label):
        """
        Assign the generated speaker labels from the base scale (the finest scale) to the longer scales.
        This process is needed to get the cluster labels for each scale. The cluster labels are needed to
        calculate the cluster-average speaker embedding for each scale.

        Args:
            uniq_id (str):
                Unique sample ID for training.
            base_scale_clus_label (torch.tensor):
                Tensor variable containing the speaker labels for the base-scale segments.
        
        Returns:
            per_scale_clus_label (torch.tensor):
                Tensor variable containing the speaker labels for each segment in each scale.
                Note that the total length of the speaker label sequence differs over scale since
                each scale has a different number of segments for the same session.

            scale_mapping (torch.tensor):
                Matrix containing the segment indices of each scale. scale_mapping is necessary for reshaping the
                multiscale embeddings to form an input matrix for the MSDD model.
        """
        per_scale_clus_label = []
        self.scale_n = len(self.multiscale_timestamp_dict[uniq_id]['scale_dict'])
        uniq_scale_mapping = get_scale_mapping_list(self.multiscale_timestamp_dict[uniq_id])
        for scale_index in range(self.scale_n):
            new_clus_label = []
            scale_seq_len = len(self.multiscale_timestamp_dict[uniq_id]["scale_dict"][scale_index]["time_stamps"])
            for seg_idx in range(scale_seq_len):
                if seg_idx in uniq_scale_mapping[scale_index]:
                    seg_clus_label = mode(base_scale_clus_label[uniq_scale_mapping[scale_index] == seg_idx])
                else:
                    seg_clus_label = 0 if len(new_clus_label) == 0 else new_clus_label[-1]
                new_clus_label.append(seg_clus_label)
            per_scale_clus_label.extend(new_clus_label)
        per_scale_clus_label = torch.tensor(per_scale_clus_label)
        return per_scale_clus_label, uniq_scale_mapping

    def get_diar_target_labels(self, uniq_id, sample, fr_level_target):
        """
        Convert frame-level diarization target variable into segment-level target variable. Since the granularity is reduced
        from frame level (10ms) to segment level (100ms~500ms), we need a threshold value, `soft_label_thres`, which determines
        the label of each segment based on the overlap between a segment range (start and end time) and the frame-level target variable.

        Args:
            uniq_id (str):
                Unique file ID that refers to an input audio file and corresponding RTTM (Annotation) file.
            sample:
                `DiarizationSpeechLabel` instance containing sample information such as audio filepath and RTTM filepath.
            fr_level_target (torch.tensor):
                Tensor containing label for each feature-level frame.

        Returns:
            seg_target (torch.tensor):
                Tensor containing binary speaker labels for base-scale segments.
            base_clus_label (torch.tensor):
                Representative speaker label for each segment. This variable only has one speaker label for each base-scale segment.
                -1 means that there is no corresponding speaker in the target_spks tuple.
        """
        seg_target_list, base_clus_label = [], []
        self.scale_n = len(self.multiscale_timestamp_dict[uniq_id]['scale_dict'])
        subseg_time_stamp_list = self.multiscale_timestamp_dict[uniq_id]["scale_dict"][self.scale_n - 1]["time_stamps"]
        for (seg_stt, seg_end) in subseg_time_stamp_list:
            seg_stt_fr, seg_end_fr = int(seg_stt * self.frame_per_sec), int(seg_end * self.frame_per_sec)
            soft_label_vec_sess = torch.sum(fr_level_target[seg_stt_fr:seg_end_fr, :], axis=0) / (
                seg_end_fr - seg_stt_fr
            )
            label_int_sess = torch.argmax(soft_label_vec_sess)
            soft_label_vec = soft_label_vec_sess.unsqueeze(0)[:, sample.target_spks].squeeze()
            if label_int_sess in sample.target_spks and torch.sum(soft_label_vec_sess) > 0:
                label_int = sample.target_spks.index(label_int_sess)
            else:
                label_int = -1
            label_vec = (soft_label_vec > self.soft_label_thres).float()
            seg_target_list.append(label_vec.detach())
            base_clus_label.append(label_int)
        seg_target = torch.stack(seg_target_list)
        base_clus_label = torch.tensor(base_clus_label)
        return seg_target, base_clus_label

    def parse_rttm_for_ms_targets(self, sample):
        """
        Generate target tensor variable by extracting groundtruth diarization labels from an RTTM file.
        This function converts (start, end, speaker_id) format into base-scale (the finest scale) segment level
        diarization label in a matrix form.

        Example of seg_target:
            [[0., 1.], [0., 1.], [1., 1.], [1., 0.], [1., 0.], ..., [0., 1.]]

        Args:
            sample:
                `DiarizationSpeechLabel` instance containing sample information such as audio filepath and RTTM filepath.
            target_spks (tuple):
                Speaker indices that are generated from combinations. If there are only one or two speakers,
                only a single target_spks tuple is generated.

        Returns:
            clus_label_index (torch.tensor):
                Groundtruth clustering label (cluster index for each segment) from RTTM files for training purpose.
            seg_target  (torch.tensor):
                Tensor variable containing hard-labels of speaker activity in each base-scale segment.
            scale_mapping (torch.tensor):
                Matrix containing the segment indices of each scale. scale_mapping is necessary for reshaping the
                multiscale embeddings to form an input matrix for the MSDD model.

        """
        rttm_lines = open(sample.rttm_file).readlines()
        uniq_id = self.get_uniq_id_with_range(sample)
        rttm_timestamps = extract_seg_info_from_rttm(uniq_id, rttm_lines)
        fr_level_target = assign_frame_level_spk_vector(
            rttm_timestamps, self.round_digits, self.frame_per_sec, target_spks=sample.target_spks
        )
        seg_target, base_clus_label = self.get_diar_target_labels(uniq_id, sample, fr_level_target)
        clus_label_index, scale_mapping = self.assign_labels_to_longer_segs(uniq_id, base_clus_label)
        return clus_label_index, seg_target, scale_mapping

    def get_uniq_id_with_range(self, sample, deci=3):
        """
        Generate unique training sample ID from unique file ID, offset and duration. The start-end time added
        unique ID is required for identifying the sample since multiple short audio samples are generated from a single
        audio file. The start time and end time of the audio stream uses millisecond units if `deci=3`.

        Args:
            sample:
                `DiarizationSpeechLabel` instance from collections.

        Returns:
            uniq_id (str):
                Unique sample ID which includes start and end time of the audio stream.
                Example: abc1001_3122_6458

        """
        bare_uniq_id = os.path.splitext(os.path.basename(sample.rttm_file))[0]
        offset = str(int(round(sample.offset, deci) * pow(10, deci)))
        endtime = str(int(round(sample.offset + sample.duration, deci) * pow(10, deci)))
        uniq_id = f"{bare_uniq_id}_{offset}_{endtime}"
        return uniq_id

    def get_ms_seg_timestamps(self, sample):
        """
        Get start and end time of segments in each scale.

        Args:
            sample:
                `DiarizationSpeechLabel` instance from preprocessing.collections
        Returns:
            ms_seg_timestamps (torch.tensor):
                Tensor containing Multiscale segment timestamps.
            ms_seg_counts (torch.tensor):
                Number of segments for each scale. This information is used for reshaping embedding batch
                during forward propagation.
        """
        uniq_id = self.get_uniq_id_with_range(sample)
        ms_seg_timestamps_list = []
        max_seq_len = len(self.multiscale_timestamp_dict[uniq_id]["scale_dict"][self.scale_n - 1]["time_stamps"])
        ms_seg_counts = [0 for _ in range(self.scale_n)]
        for scale_idx in range(self.scale_n):
            scale_ts_list = []
            for k, (seg_stt, seg_end) in enumerate(
                self.multiscale_timestamp_dict[uniq_id]["scale_dict"][scale_idx]["time_stamps"]
            ):
                stt, end = (
                    int((seg_stt - sample.offset) * self.frame_per_sec),
                    int((seg_end - sample.offset) * self.frame_per_sec),
                )
                scale_ts_list.append(torch.tensor([stt, end]).detach())
            ms_seg_counts[scale_idx] = len(
                self.multiscale_timestamp_dict[uniq_id]["scale_dict"][scale_idx]["time_stamps"]
            )
            scale_ts = torch.stack(scale_ts_list)
            scale_ts_padded = torch.cat([scale_ts, torch.zeros(max_seq_len - len(scale_ts_list), 2)], dim=0)
            ms_seg_timestamps_list.append(scale_ts_padded.detach())
        ms_seg_timestamps = torch.stack(ms_seg_timestamps_list)
        ms_seg_counts = torch.tensor(ms_seg_counts)
        return ms_seg_timestamps, ms_seg_counts

    def __getitem__(self, index):
        sample = self.collection[index]
        if sample.offset is None:
            sample.offset = 0
        clus_label_index, targets, scale_mapping = self.parse_rttm_for_ms_targets(sample)
        features = self.featurizer.process(sample.audio_file, offset=sample.offset, duration=sample.duration)
        feature_length = torch.tensor(features.shape[0]).long()
        ms_seg_timestamps, ms_seg_counts = self.get_ms_seg_timestamps(sample)
        if self.random_flip:
            torch.manual_seed(index)
            flip = torch.cat([torch.randperm(self.max_spks), torch.tensor(-1).unsqueeze(0)])
            clus_label_index, targets = flip[clus_label_index], targets[:, flip[: self.max_spks]]
        return features, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets


class _AudioMSDDInferDataset(Dataset):
    """
    Dataset class that loads a json file containing paths to audio files,
    RTTM files and number of speakers. This Dataset class is built for diarization inference and
    evaluation. Speaker embedding sequences, segment timestamps, cluster-average speaker embeddings
    are loaded from memory and fed into the dataloader.

    Example:
    {"audio_filepath": "/path/to/audio_0.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_0.rttm}
    ...
    {"audio_filepath": "/path/to/audio_n.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_n.rttm}

    Args:
        manifest_filepath (str):
             Path to input manifest json files.
        emb_dict (dict):
            Dictionary containing cluster-average embeddings and speaker mapping information.
        emb_seq (dict):
            Dictionary containing multiscale speaker embedding sequence, scale mapping and corresponding segment timestamps.
        clus_label_dict (dict):
            Subsegment-level (from base-scale) speaker labels from clustering results.
        soft_label_thres (float):
            A threshold that determines the label of each segment based on RTTM file information.
        featurizer:
            Featurizer instance for generating features from raw waveform.
        seq_eval_mode (bool):
            If True, F1 score will be calculated for each speaker pair during inference mode.
        window_stride (float):
            Window stride for acoustic feature. This value is used for calculating the numbers of feature-level frames.
        use_single_scale_clus (bool):
            Use only one scale for clustering instead of using multiple scales of embeddings for clustering.
        pairwise_infer (bool):
            This variable should be True if dataloader is created for an inference task.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        output_types = OrderedDict(
            {
                "ms_emb_seq": NeuralType(('B', 'T', 'C', 'D'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
                "ms_avg_embs": NeuralType(('B', 'C', 'D', 'C'), EncodedRepresentation()),
                "targets": NeuralType(('B', 'T', 'C'), ProbsType()),
            }
        )
        return output_types

    def __init__(
        self,
        *,
        manifest_filepath: str,
        emb_dict: Dict,
        emb_seq: Dict,
        clus_label_dict: Dict,
        soft_label_thres: float,
        seq_eval_mode: bool,
        window_stride: float,
        use_single_scale_clus: bool,
        pairwise_infer: bool,
    ):
        super().__init__()
        self.collection = DiarizationSpeechLabel(
            manifests_files=manifest_filepath.split(','),
            emb_dict=emb_dict,
            clus_label_dict=clus_label_dict,
            seq_eval_mode=seq_eval_mode,
            pairwise_infer=pairwise_infer,
        )
        self.emb_dict = emb_dict
        self.emb_seq = emb_seq
        self.clus_label_dict = clus_label_dict
        self.round_digits = 2
        self.decim = 10 ** self.round_digits
        self.frame_per_sec = int(1 / window_stride)
        self.soft_label_thres = soft_label_thres
        self.pairwise_infer = pairwise_infer
        self.max_spks = 2
        self.use_single_scale_clus = use_single_scale_clus
        self.seq_eval_mode = seq_eval_mode

    def __len__(self):
        return len(self.collection)

    def parse_rttm_multiscale(self, sample):
        """
        Generate target tensor variable by extracting groundtruth diarization labels from an RTTM file.
        This function is only used when ``self.seq_eval_mode=True`` and RTTM files are provided. This function converts
        (start, end, speaker_id) format into base-scale (the finest scale) segment level diarization label in a matrix
        form to create target matrix.

        Args:
            sample:
                DiarizationSpeechLabel instance containing sample information such as audio filepath and RTTM filepath.
            target_spks (tuple):
                Two Indices of targeted speakers for evaluation.
                Example of target_spks: (2, 3)
        Returns:
            seg_target (torch.tensor):
                Tensor variable containing hard-labels of speaker activity in each base-scale segment.
        """
        if sample.rttm_file is None:
            raise ValueError(f"RTTM file is not provided for this sample {sample}")
        rttm_lines = open(sample.rttm_file).readlines()
        uniq_id = os.path.splitext(os.path.basename(sample.rttm_file))[0]
        mapping_dict = self.emb_dict[max(self.emb_dict.keys())][uniq_id]['mapping']
        rttm_timestamps = extract_seg_info_from_rttm(uniq_id, rttm_lines, mapping_dict, sample.target_spks)
        fr_level_target = assign_frame_level_spk_vector(
            rttm_timestamps, self.round_digits, self.frame_per_sec, sample.target_spks
        )
        seg_target = self.get_diar_target_labels_from_fr_target(uniq_id, fr_level_target)
        return seg_target

    def get_diar_target_labels_from_fr_target(self, uniq_id, fr_level_target):
        """
        Generate base-scale level binary diarization label from frame-level target matrix. For the given frame-level
        speaker target matrix fr_level_target, we count the number of frames that belong to each speaker and calculate
        ratios for each speaker into the `soft_label_vec` variable. Finally, `soft_label_vec` variable is compared with `soft_label_thres`
        to determine whether a label vector should contain 0 or 1 for each speaker bin. Note that seg_target variable has
        dimension of (number of base-scale segments x 2) dimension.

        Example of seg_target:
            [[0., 1.], [0., 1.], [1., 1.], [1., 0.], [1., 0.], ..., [0., 1.]]

        Args:
            uniq_id (str):
                Unique file ID that refers to an input audio file and corresponding RTTM (Annotation) file.
            fr_level_target (torch.tensor):
                frame-level binary speaker annotation (1: exist 0: non-exist) generated from RTTM file.

        Returns:
            seg_target (torch.tensor):
                Tensor variable containing binary hard-labels of speaker activity in each base-scale segment.

        """
        if fr_level_target is None:
            return None
        else:
            seg_target_list = []
            for (seg_stt, seg_end, label_int) in self.clus_label_dict[uniq_id]:
                seg_stt_fr, seg_end_fr = int(seg_stt * self.frame_per_sec), int(seg_end * self.frame_per_sec)
                soft_label_vec = torch.sum(fr_level_target[seg_stt_fr:seg_end_fr, :], axis=0) / (
                    seg_end_fr - seg_stt_fr
                )
                label_vec = (soft_label_vec > self.soft_label_thres).int()
                seg_target_list.append(label_vec)
            seg_target = torch.stack(seg_target_list)
            return seg_target

    def __getitem__(self, index):
        sample = self.collection[index]
        if sample.offset is None:
            sample.offset = 0

        uniq_id = os.path.splitext(os.path.basename(sample.audio_file))[0]
        scale_n = len(self.emb_dict.keys())
        _avg_embs = torch.stack([self.emb_dict[scale_index][uniq_id]['avg_embs'] for scale_index in range(scale_n)])

        if self.pairwise_infer:
            avg_embs = _avg_embs[:, :, self.collection[index].target_spks]
        else:
            avg_embs = _avg_embs

        if avg_embs.shape[2] > self.max_spks:
            raise ValueError(
                f" avg_embs.shape[2] {avg_embs.shape[2]} should be less than or equal to self.max_num_speakers {self.max_spks}"
            )

        feats = []
        for scale_index in range(scale_n):
            repeat_mat = self.emb_seq["session_scale_mapping"][uniq_id][scale_index]
            feats.append(self.emb_seq[scale_index][uniq_id][repeat_mat, :])
        feats_out = torch.stack(feats).permute(1, 0, 2)
        feats_len = feats_out.shape[0]

        if self.seq_eval_mode:
            targets = self.parse_rttm_multiscale(sample)
        else:
            targets = torch.zeros(feats_len, 2).float()

        return feats_out, feats_len, targets, avg_embs


def _msdd_train_collate_fn(self, batch):
    """
    Collate batch of variables that are needed for raw waveform to diarization label training.
    The following variables are included in training/validation batch:

    Args:
        batch (tuple):
            Batch tuple containing the variables for the diarization training.
    Returns:
        features (torch.tensor):
            Raw waveform samples (time series) loaded from the audio_filepath in the input manifest file.
        feature lengths (time series sample length):
            A list of lengths of the raw waveform samples.
        ms_seg_timestamps (torch.tensor):
            Matrix containing the start time and end time (timestamps) for each segment and each scale.
            ms_seg_timestamps is needed for extracting acoustic features from raw waveforms.
        ms_seg_counts (torch.tensor):
            Matrix containing The number of segments for each scale. ms_seg_counts is necessary for reshaping
            the input matrix for the MSDD model.
        clus_label_index (torch.tensor):
            Groundtruth Clustering label (cluster index for each segment) from RTTM files for training purpose.
            clus_label_index is necessary for calculating cluster-average embedding.
        scale_mapping (torch.tensor):
            Matrix containing the segment indices of each scale. scale_mapping is necessary for reshaping the
            multiscale embeddings to form an input matrix for the MSDD model.
        targets (torch.tensor):
            Groundtruth Speaker label for the given input embedding sequence.
    """
    packed_batch = list(zip(*batch))
    features, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets = packed_batch
    features_list, feature_length_list = [], []
    ms_seg_timestamps_list, ms_seg_counts_list, scale_clus_label_list, scale_mapping_list, targets_list = (
        [],
        [],
        [],
        [],
        [],
    )

    max_raw_feat_len = max([x.shape[0] for x in features])
    max_target_len = max([x.shape[0] for x in targets])
    max_total_seg_len = max([x.shape[0] for x in clus_label_index])

    for feat, feat_len, ms_seg_ts, ms_seg_ct, scale_clus, scl_map, tgt in batch:
        seq_len = tgt.shape[0]
        pad_feat = (0, max_raw_feat_len - feat_len)
        pad_tgt = (0, 0, 0, max_target_len - seq_len)
        pad_sm = (0, max_target_len - seq_len)
        pad_ts = (0, 0, 0, max_target_len - seq_len)
        pad_sc = (0, max_total_seg_len - scale_clus.shape[0])
        padded_feat = torch.nn.functional.pad(feat, pad_feat)
        padded_tgt = torch.nn.functional.pad(tgt, pad_tgt)
        padded_sm = torch.nn.functional.pad(scl_map, pad_sm)
        padded_ms_seg_ts = torch.nn.functional.pad(ms_seg_ts, pad_ts)
        padded_scale_clus = torch.nn.functional.pad(scale_clus, pad_sc)

        features_list.append(padded_feat)
        feature_length_list.append(feat_len.clone().detach())
        ms_seg_timestamps_list.append(padded_ms_seg_ts)
        ms_seg_counts_list.append(ms_seg_ct.clone().detach())
        scale_clus_label_list.append(padded_scale_clus)
        scale_mapping_list.append(padded_sm)
        targets_list.append(padded_tgt)

    features = torch.stack(features_list)
    feature_length = torch.stack(feature_length_list)
    ms_seg_timestamps = torch.stack(ms_seg_timestamps_list)
    clus_label_index = torch.stack(scale_clus_label_list)
    ms_seg_counts = torch.stack(ms_seg_counts_list)
    scale_mapping = torch.stack(scale_mapping_list)
    targets = torch.stack(targets_list)
    return features, feature_length, ms_seg_timestamps, ms_seg_counts, clus_label_index, scale_mapping, targets


def _msdd_infer_collate_fn(self, batch):
    """
    Collate batch of feats (speaker embeddings), feature lengths, target label sequences and cluster-average embeddings.

    Args:
        batch (tuple):
            Batch tuple containing feats, feats_len, targets and ms_avg_embs.
    Returns:
        feats (torch.tensor):
            Collated speaker embedding with unified length.
        feats_len (torch.tensor):
            The actual length of each embedding sequence without zero padding.
        targets (torch.tensor):
            Groundtruth Speaker label for the given input embedding sequence.
        ms_avg_embs (torch.tensor):
            Cluster-average speaker embedding vectors.
    """

    packed_batch = list(zip(*batch))
    feats, feats_len, targets, ms_avg_embs = packed_batch
    feats_list, flen_list, targets_list, ms_avg_embs_list = [], [], [], []
    max_audio_len = max(feats_len)
    max_target_len = max([x.shape[0] for x in targets])

    for feature, feat_len, target, ivector in batch:
        flen_list.append(feat_len)
        ms_avg_embs_list.append(ivector)
        if feat_len < max_audio_len:
            pad_a = (0, 0, 0, 0, 0, max_audio_len - feat_len)
            pad_t = (0, 0, 0, max_target_len - target.shape[0])
            padded_feature = torch.nn.functional.pad(feature, pad_a)
            padded_target = torch.nn.functional.pad(target, pad_t)
            feats_list.append(padded_feature)
            targets_list.append(padded_target)
        else:
            targets_list.append(target.clone().detach())
            feats_list.append(feature.clone().detach())

    feats = torch.stack(feats_list)
    feats_len = torch.tensor(flen_list)
    targets = torch.stack(targets_list)
    ms_avg_embs = torch.stack(ms_avg_embs_list)
    return feats, feats_len, targets, ms_avg_embs


class AudioToSpeechMSDDTrainDataset(_AudioMSDDTrainDataset):
    """
    Dataset class that loads a json file containing paths to audio files,
    rttm files and number of speakers. This Dataset class is designed for
    training or fine-tuning speaker embedding extractor and diarization decoder
    at the same time.

    Example:
    {"audio_filepath": "/path/to/audio_0.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_0.rttm}
    ...
    {"audio_filepath": "/path/to/audio_n.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_n.rttm}

    Args:
        manifest_filepath (str):
            Path to input manifest json files.
        multiscale_args_dict (dict):
            Dictionary containing the parameters for multiscale segmentation and clustering.
        emb_dir (str):
            Path to a temporary folder where segmentation information for embedding extraction is saved.
        soft_label_thres (float):
            A threshold that determines the label of each segment based on RTTM file information.
        featurizer:
            Featurizer instance for generating features from the raw waveform.
        window_stride (float):
            Window stride for acoustic feature. This value is used for calculating the numbers of feature-level frames.
        emb_batch_size (int):
            Number of embedding vectors that are trained with attached computational graphs.
        pairwise_infer (bool):
            This variable should be True if dataloader is created for an inference task.
    """

    def __init__(
        self,
        *,
        manifest_filepath: str,
        multiscale_args_dict: Dict,
        emb_dir: str,
        soft_label_thres: float,
        featurizer,
        window_stride,
        emb_batch_size,
        pairwise_infer: bool,
        global_rank: int,
    ):
        super().__init__(
            manifest_filepath=manifest_filepath,
            multiscale_args_dict=multiscale_args_dict,
            emb_dir=emb_dir,
            soft_label_thres=soft_label_thres,
            featurizer=featurizer,
            window_stride=window_stride,
            emb_batch_size=emb_batch_size,
            pairwise_infer=pairwise_infer,
            global_rank=global_rank,
        )

    def msdd_train_collate_fn(self, batch):
        return _msdd_train_collate_fn(self, batch)


class AudioToSpeechMSDDInferDataset(_AudioMSDDInferDataset):
    """
    Dataset class that loads a json file containing paths to audio files,
    rttm files and number of speakers. The created labels are used for diarization inference.

    Example:
    {"audio_filepath": "/path/to/audio_0.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_0.rttm}
    ...
    {"audio_filepath": "/path/to/audio_n.wav", "num_speakers": 2,
    "rttm_filepath": "/path/to/diar_label_n.rttm}

    Args:
        manifest_filepath (str):
            Path to input manifest json files.
        emb_dict (dict):
            Dictionary containing cluster-average embeddings and speaker mapping information.
        emb_seq (dict):
            Dictionary containing multiscale speaker embedding sequence, scale mapping and corresponding segment timestamps.
        clus_label_dict (dict):
            Subsegment-level (from base-scale) speaker labels from clustering results.
        soft_label_thres (float):
            Threshold that determines speaker labels of segments depending on the overlap with groundtruth speaker timestamps.
        featurizer:
            Featurizer instance for generating features from raw waveform.
        use_single_scale_clus (bool):
            Use only one scale for clustering instead of using multiple scales of embeddings for clustering.
        seq_eval_mode (bool):
            If True, F1 score will be calculated for each speaker pair during inference mode.
        window_stride (float):
            Window stride for acoustic feature. This value is used for calculating the numbers of feature-level frames.
        pairwise_infer (bool):
            If True, this Dataset class operates in inference mode. In inference mode, a set of speakers in the input audio
            is split into multiple pairs of speakers and speaker tuples (e.g. 3 speakers: [(0,1), (1,2), (0,2)]) and then
            fed into the MSDD to merge the individual results.
    """

    def __init__(
        self,
        *,
        manifest_filepath: str,
        emb_dict: Dict,
        emb_seq: Dict,
        clus_label_dict: Dict,
        soft_label_thres: float,
        use_single_scale_clus: bool,
        seq_eval_mode: bool,
        window_stride: float,
        pairwise_infer: bool,
    ):
        super().__init__(
            manifest_filepath=manifest_filepath,
            emb_dict=emb_dict,
            emb_seq=emb_seq,
            clus_label_dict=clus_label_dict,
            soft_label_thres=soft_label_thres,
            use_single_scale_clus=use_single_scale_clus,
            window_stride=window_stride,
            seq_eval_mode=seq_eval_mode,
            pairwise_infer=pairwise_infer,
        )

    def msdd_infer_collate_fn(self, batch):
        return _msdd_infer_collate_fn(self, batch)
