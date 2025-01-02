# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#
# Copyright 2018-2019, Mingkun Huang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import random
from typing import Optional, Tuple

import numba
import torch
from numba import cuda

from nemo.collections.asr.parts.numba.rnnt_loss.utils import global_constants, rnnt_helper
from nemo.collections.asr.parts.numba.rnnt_loss.utils.cuda_utils import gpu_rnnt_kernel, reduce


class GPURNNT:
    def __init__(
        self,
        minibatch: int,
        maxT: int,
        maxU: int,
        alphabet_size: int,
        workspace,
        blank: int,
        fastemit_lambda: float,
        clamp: float,
        num_threads: int,
        stream,
    ):
        """
        Helper class to launch the CUDA Kernels to compute the Transducer Loss.

        Args:
            minibatch: Int representing the batch size.
            maxT: The maximum possible acoustic sequence length. Represents T in the logprobs tensor.
            maxU: The maximum possible target sequence length. Represents U in the logprobs tensor.
            alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT blank).
            workspace: An allocated chunk of memory that will be sliced off and reshaped into required
                blocks used as working memory.
            blank: Index of the RNNT blank token in the vocabulary. Generally the first or last token in the vocab.
            fastemit_lambda: Float scaling factor for FastEmit regularization. Refer to
                FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization.
            clamp: Float value. When set to value >= 0.0, will clamp the gradient to [-clamp, clamp].
            num_threads: Number of OMP threads to launch.
            stream: Numba Cuda Stream.
        """
        self.minibatch_ = minibatch
        self.maxT_ = maxT
        self.maxU_ = maxU
        self.alphabet_size_ = alphabet_size
        self.gpu_workspace = cuda.as_cuda_array(
            workspace
        )  # a flat vector of floatX numbers that represents allocated memory slices
        self.blank_ = blank
        self.fastemit_lambda_ = fastemit_lambda
        self.clamp_ = abs(clamp)
        self.num_threads_ = num_threads
        self.stream_ = stream  # type: cuda.cudadrv.driver.Stream

        _torch_num_threads = torch.get_num_threads()
        if num_threads > 0:
            numba.set_num_threads(min(multiprocessing.cpu_count(), num_threads))
            self.num_threads_ = numba.get_num_threads()
        else:
            self.num_threads_ = numba.get_num_threads()
        torch.set_num_threads(_torch_num_threads)

    def log_softmax(self, acts: torch.Tensor, denom: torch.Tensor):
        """
        Computes the log softmax denominator of the input activation tensor
        and stores the result in denom.

        Args:
            acts: Activation tensor of shape [B, T, U, V+1]. The input must be represented as a flat tensor
                of shape [B * T * U * (V+1)] to allow pointer indexing.
            denom: A zero tensor of same shape as acts.

        Updates:
            This kernel inplace updates the `denom` tensor
        """
        # // trans_acts + pred_acts -> log_softmax denominator
        reduce.reduce_max(
            acts,
            denom,
            rows=self.alphabet_size_,
            cols=self.minibatch_ * self.maxT_ * self.maxU_,
            minus=False,
            stream=self.stream_,
        )

        reduce.reduce_exp(
            acts,
            denom,
            rows=self.alphabet_size_,
            cols=self.minibatch_ * self.maxT_ * self.maxU_,
            minus=True,
            stream=self.stream_,
        )

    def compute_cost_and_score(
        self,
        acts: torch.Tensor,
        grads: Optional[torch.Tensor],
        costs: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> global_constants.RNNTStatus:
        """
        Compute both the loss and the gradients.

        Args:
            acts: A flattened tensor of shape [B, T, U, V+1] representing the activation matrix.
            grad: A flattented zero tensor of same shape as acts.
            costs: A zero vector of length B which will be updated inplace with the log probability costs.
            flat_labels: A flattened matrix of labels of shape [B, U]
            label_lengths: A vector of length B that contains the original lengths of the acoustic sequence.
            input_lengths: A vector of length B that contains the original lengths of the target sequence.

        Updates:
            This will launch kernels that will update inline the following variables:
            -   grads: Gradients of the activation matrix wrt the costs vector.
            -   costs: Negative log likelihood of the forward variable.

        Returns:
            An enum that either represents a successful RNNT operation or failure.
        """
        training = grads is not None

        if training:
            grads *= 0.0  # zero grads

        used_offset, (denom, alphas, betas, llForward, llBackward) = self._prepare_workspace()

        ######## START EXECUTION ########
        self.log_softmax(acts, denom)

        # Compute alphas
        gpu_rnnt_kernel.compute_alphas_kernel[self.minibatch_, self.maxU_, self.stream_, 0](
            acts,
            denom,
            alphas,
            llForward,
            input_lengths,
            label_lengths,
            labels,
            self.minibatch_,
            self.maxT_,
            self.maxU_,
            self.alphabet_size_,
            self.blank_,
        )

        if training:
            # Compute betas
            gpu_rnnt_kernel.compute_betas_kernel[self.minibatch_, self.maxU_, self.stream_, 0](
                acts,
                denom,
                betas,
                llBackward,
                input_lengths,
                label_lengths,
                labels,
                self.minibatch_,
                self.maxT_,
                self.maxU_,
                self.alphabet_size_,
                self.blank_,
            )

            # Compute gradient
            grad_blocks_per_grid = self.minibatch_ * self.maxT_ * self.maxU_
            grad_threads_per_block = gpu_rnnt_kernel.GPU_RNNT_THREAD_SIZE
            gpu_rnnt_kernel.compute_grad_kernel[grad_blocks_per_grid, grad_threads_per_block, self.stream_, 0](
                grads,
                acts,
                denom,
                alphas,
                betas,
                llForward,
                input_lengths,
                label_lengths,
                labels,
                self.minibatch_,
                self.maxT_,
                self.maxU_,
                self.alphabet_size_,
                self.blank_,
                self.fastemit_lambda_,
                self.clamp_,
            )

        # // cost copy, negate (for log likelihood) and update with additional regularizers
        # This needs to be done via CUDA, because we used temporary memory llForward
        # passed to alpha, which was updated with log likelihoods.
        # But copying this data into a pytorch pointer is more difficult (numba api is one way)
        # Therefore launch a pointwise CUDA kernel to update the costs inplace from data of llForward
        # Then negate to compute the loglikelihood.
        threadsperblock = min(costs.shape[0], 32)
        blockspergrid = (costs.shape[0] + (threadsperblock - 1)) // threadsperblock
        rnnt_helper.compute_costs_data[blockspergrid, threadsperblock, self.stream_, 0](
            llForward, costs, self.fastemit_lambda_
        )
        self.stream_.synchronize()

        return global_constants.RNNTStatus.RNNT_STATUS_SUCCESS

    def cost_and_grad(
        self,
        acts: torch.Tensor,
        grads: torch.Tensor,
        costs: torch.Tensor,
        pad_labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        if (
            acts is None
            or grads is None
            or costs is None
            or pad_labels is None
            or label_lengths is None
            or input_lengths is None
        ):
            return global_constants.RNNTStatus.RNNT_STATUS_INVALID_VALUE

        return self.compute_cost_and_score(acts, grads, costs, pad_labels, label_lengths, input_lengths)

    def score_forward(
        self,
        acts: torch.Tensor,
        costs: torch.Tensor,
        pad_labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        if acts is None or costs is None or pad_labels is None or label_lengths is None or input_lengths is None:
            return global_constants.RNNTStatus.RNNT_STATUS_INVALID_VALUE

        return self.compute_cost_and_score(acts, None, costs, pad_labels, label_lengths, input_lengths)

    def _prepare_workspace(self) -> Tuple[int, Tuple[torch.Tensor, ...]]:
        """
        Helper method that uses the workspace and constructs slices of it that can be used.

        Returns:
            An int, representing the offset of the used workspace (practically, the slice of the workspace consumed)
            A tuple of tensors representing the shared workspace.
        """
        used_offset = 0

        # // denom
        denom = self.gpu_workspace[used_offset : used_offset + self.maxT_ * self.maxU_ * self.minibatch_]
        used_offset += self.maxT_ * self.maxU_ * self.minibatch_

        # // alphas & betas
        alphas = self.gpu_workspace[used_offset : used_offset + self.maxT_ * self.maxU_ * self.minibatch_]
        used_offset += self.maxT_ * self.maxU_ * self.minibatch_
        betas = self.gpu_workspace[used_offset : used_offset + self.maxT_ * self.maxU_ * self.minibatch_]
        used_offset += self.maxT_ * self.maxU_ * self.minibatch_

        # // logllh
        llForward = self.gpu_workspace[used_offset : used_offset + self.minibatch_]
        used_offset += self.minibatch_
        llBackward = self.gpu_workspace[used_offset : used_offset + self.minibatch_]
        used_offset += self.minibatch_

        return used_offset, (denom, alphas, betas, llForward, llBackward)


class MultiblankGPURNNT(GPURNNT):
    def __init__(
        self,
        sigma: float,
        num_big_blanks: int,
        minibatch: int,
        maxT: int,
        maxU: int,
        alphabet_size: int,
        workspace,
        big_blank_workspace,
        blank: int,
        fastemit_lambda: float,
        clamp: float,
        num_threads: int,
        stream,
    ):
        """
        Helper class to launch the CUDA Kernels to compute Multi-blank Transducer Loss (https://arxiv.org/pdf/2211.03541).

        Args:
            sigma: Hyper-parameter related to the logit-normalization method in training multi-blank transducers.
            num_big_blanks: Number of big blank symbols the model has. This should not include the standard blank symbol.
            minibatch: Int representing the batch size.
            maxT: The maximum possible acoustic sequence length. Represents T in the logprobs tensor.
            maxU: The maximum possible target sequence length. Represents U in the logprobs tensor.
            alphabet_size: The vocabulary dimension V + 1 + num-big-blanks
            workspace: An allocated chunk of memory that will be sliced off and reshaped into required
                blocks used as working memory.
            big_blank_workspace: An allocated chunk of memory that will be sliced off and reshaped into required
                blocks used as working memory specifically for the multi-blank related computations.
            blank: Index of the RNNT blank token in the vocabulary. Generally the first or last token in the vocab.
            fastemit_lambda: Float scaling factor for FastEmit regularization. Refer to
                FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization.
            clamp: Float value. When set to value >= 0.0, will clamp the gradient to [-clamp, clamp].
            num_threads: Number of OMP threads to launch.
            stream: Numba Cuda Stream.
        """
        super().__init__(
            minibatch, maxT, maxU, alphabet_size, workspace, blank, fastemit_lambda, clamp, num_threads, stream
        )
        self.big_blank_workspace = cuda.as_cuda_array(
            big_blank_workspace
        )  # a flat vector of integer numbers that represents allocated memory slices

        self.num_big_blanks = num_big_blanks
        self.sigma = sigma

    def compute_cost_and_score(
        self,
        acts: torch.Tensor,
        grads: Optional[torch.Tensor],
        costs: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> global_constants.RNNTStatus:
        """
        Compute both the loss and the gradients.

        Args:
            acts: A flattened tensor of shape [B, T, U, V+1] representing the activation matrix.
            grad: A flattented zero tensor of same shape as acts.
            costs: A zero vector of length B which will be updated inplace with the log probability costs.
            flat_labels: A flattened matrix of labels of shape [B, U]
            label_lengths: A vector of length B that contains the original lengths of the acoustic sequence.
            input_lengths: A vector of length B that contains the original lengths of the target sequence.

        Updates:
            This will launch kernels that will update inline the following variables:
            -   grads: Gradients of the activation matrix wrt the costs vector.
            -   costs: Negative log likelihood of the forward variable.

        Returns:
            An enum that either represents a successful RNNT operation or failure.
        """
        training = grads is not None

        if training:
            grads *= 0.0  # zero grads

        _, (denom, alphas, betas, llForward, llBackward, bigblank_durations) = self._prepare_workspace()

        ######## START EXECUTION ########
        self.log_softmax(acts, denom)

        # Compute alphas
        gpu_rnnt_kernel.compute_multiblank_alphas_kernel[self.minibatch_, self.maxU_, self.stream_, 0](
            acts,
            denom,
            self.sigma,
            alphas,
            llForward,
            input_lengths,
            label_lengths,
            labels,
            self.minibatch_,
            self.maxT_,
            self.maxU_,
            self.alphabet_size_,
            self.blank_,
            bigblank_durations,
            self.num_big_blanks,
        )

        if training:
            # Compute betas
            gpu_rnnt_kernel.compute_multiblank_betas_kernel[self.minibatch_, self.maxU_, self.stream_, 0](
                acts,
                denom,
                self.sigma,
                betas,
                llBackward,
                input_lengths,
                label_lengths,
                labels,
                self.minibatch_,
                self.maxT_,
                self.maxU_,
                self.alphabet_size_,
                self.blank_,
                bigblank_durations,
                self.num_big_blanks,
            )

            # Compute gradient
            grad_blocks_per_grid = self.minibatch_ * self.maxT_ * self.maxU_
            grad_threads_per_block = gpu_rnnt_kernel.GPU_RNNT_THREAD_SIZE
            gpu_rnnt_kernel.compute_multiblank_grad_kernel[
                grad_blocks_per_grid, grad_threads_per_block, self.stream_, 0
            ](
                grads,
                acts,
                denom,
                self.sigma,
                alphas,
                betas,
                llForward,
                input_lengths,
                label_lengths,
                labels,
                self.minibatch_,
                self.maxT_,
                self.maxU_,
                self.alphabet_size_,
                self.blank_,
                bigblank_durations,
                self.num_big_blanks,
                self.fastemit_lambda_,
                self.clamp_,
            )

        # // cost copy, negate (for log likelihood) and update with additional regularizers
        # This needs to be done via CUDA, because we used temporary memory llForward
        # passed to alpha, which was updated with log likelihoods.
        # But copying this data into a pytorch pointer is more difficult (numba api is one way)
        # Therefore launch a pointwise CUDA kernel to update the costs inplace from data of llForward
        # Then negate to compute the loglikelihood.
        threadsperblock = min(costs.shape[0], 32)
        blockspergrid = (costs.shape[0] + (threadsperblock - 1)) // threadsperblock
        rnnt_helper.compute_costs_data[blockspergrid, threadsperblock, self.stream_, 0](
            llForward, costs, self.fastemit_lambda_
        )
        self.stream_.synchronize()

        return global_constants.RNNTStatus.RNNT_STATUS_SUCCESS

    def cost_and_grad(
        self,
        acts: torch.Tensor,
        grads: torch.Tensor,
        costs: torch.Tensor,
        pad_labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        if (
            acts is None
            or grads is None
            or costs is None
            or pad_labels is None
            or label_lengths is None
            or input_lengths is None
        ):
            return global_constants.RNNTStatus.RNNT_STATUS_INVALID_VALUE

        return self.compute_cost_and_score(acts, grads, costs, pad_labels, label_lengths, input_lengths)

    def score_forward(
        self,
        acts: torch.Tensor,
        costs: torch.Tensor,
        pad_labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        if acts is None or costs is None or pad_labels is None or label_lengths is None or input_lengths is None:
            return global_constants.RNNTStatus.RNNT_STATUS_INVALID_VALUE

        return self.compute_cost_and_score(acts, None, costs, pad_labels, label_lengths, input_lengths)

    def _prepare_workspace(self) -> (int, Tuple[torch.Tensor]):
        """
        Helper method that uses the workspace and constructs slices of it that can be used.

        Returns:
            An int, representing the offset of the used workspace (practically, the slice of the workspace consumed)
            A tuple of tensors representing the shared workspace.
        """
        used_offset, (denom, alphas, betas, llForward, llBackward) = super()._prepare_workspace()

        bigblank_durations = self.big_blank_workspace[: self.num_big_blanks]

        return used_offset, (denom, alphas, betas, llForward, llBackward, bigblank_durations)


class GPUTDT(GPURNNT):
    def __init__(
        self,
        sigma: float,
        omega: float,
        num_durations: int,
        minibatch: int,
        maxT: int,
        maxU: int,
        alphabet_size: int,
        workspace,
        tdt_workspace,
        blank: int,
        fastemit_lambda: float,
        clamp: float,
        num_threads: int,
        stream,
    ):
        """
        Helper class to launch the CUDA Kernels to compute TDT Loss (https://arxiv.org/pdf/2211.03541).

        Args:
            sigma: Hyper-parameter related to the logit-normalization method in training tdt transducers.
            omega: Hyper-parameter related to the sampled training.
            num_durations: Number of durations the model supports.
            minibatch: Int representing the batch size.
            maxT: The maximum possible acoustic sequence length. Represents T in the logprobs tensor.
            maxU: The maximum possible target sequence length. Represents U in the logprobs tensor.
            alphabet_size: The vocabulary dimension V + 1 + num-big-blanks
            workspace: An allocated chunk of memory that will be sliced off and reshaped into required
                blocks used as working memory.
            tdt_workspace: An allocated chunk of memory that will be sliced off and reshaped into required
                blocks used as working memory specifically for the tdt related computations.
            blank: Index of the blank token in the vocabulary. Must be the last token in the vocab.
            fastemit_lambda: Float scaling factor for FastEmit regularization. Refer to
                FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization.
            clamp: Float value. When set to value >= 0.0, will clamp the gradient to [-clamp, clamp].
            num_threads: Number of OMP threads to launch.
            stream: Numba Cuda Stream.
        """
        super().__init__(
            minibatch, maxT, maxU, alphabet_size, workspace, blank, fastemit_lambda, clamp, num_threads, stream
        )
        self.tdt_workspace = cuda.as_cuda_array(
            tdt_workspace
        )  # a flat vector of integer numbers that represents allocated memory slices

        self.num_durations = num_durations
        self.sigma = sigma
        self.omega = omega

    def compute_cost_and_score(
        self,
        label_acts: torch.Tensor,
        duration_acts: torch.Tensor,
        label_grads: Optional[torch.Tensor],
        duration_grads: Optional[torch.Tensor],
        costs: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> global_constants.RNNTStatus:
        """
        Compute both the loss and the gradients.

        Args:
            label_acts: A flattened tensor of shape [B, T, U, V] representing the activation matrix for tokens.
            duration_acts: A flattened tensor of shape [B, T, U, D] representing the activation matrix for durations.
            label_grad: A flattented zero tensor of same shape as label_acts.
            duration_grad: A flattented zero tensor of same shape as duration_acts.
            costs: A zero vector of length B which will be updated inplace with the log probability costs.
            flat_labels: A flattened matrix of labels of shape [B, U]
            label_lengths: A vector of length B that contains the original lengths of the acoustic sequence.
            input_lengths: A vector of length B that contains the original lengths of the target sequence.

        Updates:
            This will launch kernels that will update inline the following variables:
            -   *_grads: Gradients of the activation matrix wrt the costs vector.
            -   costs: Negative log likelihood of the forward variable.

        Returns:
            An enum that either represents a successful RNNT operation or failure.
        """
        training = label_grads is not None

        if training:
            label_grads *= 0.0  # zero grads
            duration_grads *= 0.0  # zero grads

        _, (denom, alphas, betas, llForward, llBackward, durations) = self._prepare_workspace()

        ######## START EXECUTION ########
        self.log_softmax(label_acts, denom)

        r = random.uniform(0, 1)
        if r < self.omega:
            # Compute alphas
            gpu_rnnt_kernel.compute_alphas_kernel[self.minibatch_, self.maxU_, self.stream_, 0](
                label_acts,
                denom,
                alphas,
                llForward,
                input_lengths,
                label_lengths,
                labels,
                self.minibatch_,
                self.maxT_,
                self.maxU_,
                self.alphabet_size_,
                self.blank_,
            )
        else:
            # Compute alphas
            gpu_rnnt_kernel.compute_tdt_alphas_kernel[self.minibatch_, self.maxU_, self.stream_, 0](
                label_acts,
                duration_acts,
                denom,
                self.sigma,
                alphas,
                llForward,
                input_lengths,
                label_lengths,
                labels,
                self.minibatch_,
                self.maxT_,
                self.maxU_,
                self.alphabet_size_,
                self.blank_,
                durations,
                self.num_durations,
            )

        if training:
            # Compute betas
            if r < self.omega:
                gpu_rnnt_kernel.compute_betas_kernel[self.minibatch_, self.maxU_, self.stream_, 0](
                    label_acts,
                    denom,
                    betas,
                    llBackward,
                    input_lengths,
                    label_lengths,
                    labels,
                    self.minibatch_,
                    self.maxT_,
                    self.maxU_,
                    self.alphabet_size_,
                    self.blank_,
                )

                # Compute gradient
                grad_blocks_per_grid = self.minibatch_ * self.maxT_ * self.maxU_
                grad_threads_per_block = gpu_rnnt_kernel.GPU_RNNT_THREAD_SIZE
                gpu_rnnt_kernel.compute_grad_kernel[grad_blocks_per_grid, grad_threads_per_block, self.stream_, 0](
                    label_grads,
                    label_acts,
                    denom,
                    alphas,
                    betas,
                    llForward,
                    input_lengths,
                    label_lengths,
                    labels,
                    self.minibatch_,
                    self.maxT_,
                    self.maxU_,
                    self.alphabet_size_,
                    self.blank_,
                    self.fastemit_lambda_,
                    self.clamp_,
                )
            else:
                gpu_rnnt_kernel.compute_tdt_betas_kernel[self.minibatch_, self.maxU_, self.stream_, 0](
                    label_acts,
                    duration_acts,
                    denom,
                    self.sigma,
                    betas,
                    llBackward,
                    input_lengths,
                    label_lengths,
                    labels,
                    self.minibatch_,
                    self.maxT_,
                    self.maxU_,
                    self.alphabet_size_,
                    self.blank_,
                    durations,
                    self.num_durations,
                )

                # Compute gradient
                grad_blocks_per_grid = self.minibatch_ * self.maxT_ * self.maxU_
                grad_threads_per_block = gpu_rnnt_kernel.GPU_RNNT_THREAD_SIZE
                gpu_rnnt_kernel.compute_tdt_grad_kernel[grad_blocks_per_grid, grad_threads_per_block, self.stream_, 0](
                    label_grads,
                    duration_grads,
                    label_acts,
                    duration_acts,
                    denom,
                    self.sigma,
                    alphas,
                    betas,
                    llForward,
                    input_lengths,
                    label_lengths,
                    labels,
                    self.minibatch_,
                    self.maxT_,
                    self.maxU_,
                    self.alphabet_size_,
                    self.blank_,
                    durations,
                    self.num_durations,
                    self.fastemit_lambda_,
                    self.clamp_,
                )

        # // cost copy, negate (for log likelihood) and update with additional regularizers
        # This needs to be done via CUDA, because we used temporary memory llForward
        # passed to alpha, which was updated with log likelihoods.
        # But copying this data into a pytorch pointer is more difficult (numba api is one way)
        # Therefore launch a pointwise CUDA kernel to update the costs inplace from data of llForward
        # Then negate to compute the loglikelihood.
        threadsperblock = min(costs.shape[0], 32)
        blockspergrid = (costs.shape[0] + (threadsperblock - 1)) // threadsperblock
        rnnt_helper.compute_costs_data[blockspergrid, threadsperblock, self.stream_, 0](
            llForward, costs, self.fastemit_lambda_
        )
        self.stream_.synchronize()

        return global_constants.RNNTStatus.RNNT_STATUS_SUCCESS

    def cost_and_grad(
        self,
        label_acts: torch.Tensor,
        duration_acts: torch.Tensor,
        label_grads: torch.Tensor,
        duration_grads: torch.Tensor,
        costs: torch.Tensor,
        pad_labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        if (
            duration_acts is None
            or label_acts is None
            or label_grads is None
            or duration_grads is None
            or costs is None
            or pad_labels is None
            or label_lengths is None
            or input_lengths is None
        ):
            return global_constants.RNNTStatus.RNNT_STATUS_INVALID_VALUE

        return self.compute_cost_and_score(
            label_acts, duration_acts, label_grads, duration_grads, costs, pad_labels, label_lengths, input_lengths
        )

    def score_forward(
        self,
        label_acts: torch.Tensor,
        duration_acts: torch.Tensor,
        costs: torch.Tensor,
        pad_labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        if (
            label_acts is None
            or duration_acts is None
            or costs is None
            or pad_labels is None
            or label_lengths is None
            or input_lengths is None
        ):
            return global_constants.RNNTStatus.RNNT_STATUS_INVALID_VALUE

        return self.compute_cost_and_score(
            label_acts, duration_acts, None, None, costs, pad_labels, label_lengths, input_lengths
        )

    def _prepare_workspace(self) -> (int, Tuple[torch.Tensor]):
        """
        Helper method that uses the workspace and constructs slices of it that can be used.

        Returns:
            An int, representing the offset of the used workspace (practically, the slice of the workspace consumed)
            A tuple of tensors representing the shared workspace.
        """
        used_offset, (denom, alphas, betas, llForward, llBackward) = super()._prepare_workspace()

        durations = self.tdt_workspace[: self.num_durations]

        return used_offset, (denom, alphas, betas, llForward, llBackward, durations)
