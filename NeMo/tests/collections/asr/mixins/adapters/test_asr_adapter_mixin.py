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

import pytest
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from nemo.collections.asr.models import ASRModel, EncDecCTCModel, EncDecRNNTModel
from nemo.collections.asr.parts.submodules.adapters import multi_head_attention_adapter_module
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.collections.common.parts import adapter_modules
from nemo.core.classes.mixins.access_mixins import AccessMixin
from nemo.core.classes.mixins.adapter_mixins import AdapterModuleMixin, get_registered_adapter
from nemo.core.utils import numba_utils
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__
from nemo.utils import config_utils, model_utils

NUMBA_RNNT_LOSS_AVAILABLE = numba_utils.numba_cpu_is_supported(
    __NUMBA_MINIMUM_VERSION__
) or numba_utils.numba_cuda_is_supported(__NUMBA_MINIMUM_VERSION__)


@pytest.fixture()
def model():
    preprocessor = {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor'}
    encoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASREncoderAdapter',
        'feat_in': 64,
        'activation': 'relu',
        'conv_mask': True,
        'jasper': [
            {
                'filters': 50,
                'repeat': 1,
                'kernel': [1],
                'stride': [1],
                'dilation': [1],
                'dropout': 0.0,
                'residual': False,
                'separable': True,
                'se': True,
                'se_context_size': -1,
            }
        ],
    }

    decoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
        'feat_in': 50,
        'num_classes': 28,
        'vocabulary': [
            ' ',
            'a',
            'b',
            'c',
            'd',
            'e',
            'f',
            'g',
            'h',
            'i',
            'j',
            'k',
            'l',
            'm',
            'n',
            'o',
            'p',
            'q',
            'r',
            's',
            't',
            'u',
            'v',
            'w',
            'x',
            'y',
            'z',
            "'",
        ],
    }
    modelConfig = DictConfig(
        {'preprocessor': DictConfig(preprocessor), 'encoder': DictConfig(encoder), 'decoder': DictConfig(decoder)}
    )

    model_instance = EncDecCTCModel(cfg=modelConfig)
    return model_instance


@pytest.fixture()
def conformer_ctc_adapter():
    preprocessor = {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor'}
    encoder = {
        '_target_': 'nemo.collections.asr.modules.ConformerEncoderAdapter',
        'feat_in': 64,
        'feat_out': -1,
        'n_layers': 2,
        'd_model': 128,
        'subsampling': 'striding',
        'subsampling_factor': 4,
        'self_attention_model': 'rel_pos',
        'n_heads': 4,
        'conv_kernel_size': 31,
    }

    decoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
        'feat_in': 128,
        'num_classes': 28,
        'vocabulary': [
            ' ',
            'a',
            'b',
            'c',
            'd',
            'e',
            'f',
            'g',
            'h',
            'i',
            'j',
            'k',
            'l',
            'm',
            'n',
            'o',
            'p',
            'q',
            'r',
            's',
            't',
            'u',
            'v',
            'w',
            'x',
            'y',
            'z',
            "'",
        ],
    }
    modelConfig = DictConfig(
        {'preprocessor': DictConfig(preprocessor), 'encoder': DictConfig(encoder), 'decoder': DictConfig(decoder)}
    )

    model_instance = EncDecCTCModel(cfg=modelConfig)
    return model_instance


@pytest.fixture()
def squeezeformer_ctc_adapter():
    preprocessor = {'_target_': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor'}
    encoder = {
        '_target_': 'nemo.collections.asr.modules.SqueezeformerEncoderAdapter',
        'feat_in': 64,
        'feat_out': -1,
        'n_layers': 2,
        'd_model': 128,
        'time_reduce_idx': 1,
        'subsampling': 'dw_striding',
        'subsampling_factor': 4,
        'self_attention_model': 'rel_pos',
        'n_heads': 4,
        'conv_kernel_size': 31,
    }

    decoder = {
        '_target_': 'nemo.collections.asr.modules.ConvASRDecoder',
        'feat_in': 128,
        'num_classes': 28,
        'vocabulary': [
            ' ',
            'a',
            'b',
            'c',
            'd',
            'e',
            'f',
            'g',
            'h',
            'i',
            'j',
            'k',
            'l',
            'm',
            'n',
            'o',
            'p',
            'q',
            'r',
            's',
            't',
            'u',
            'v',
            'w',
            'x',
            'y',
            'z',
            "'",
        ],
    }
    modelConfig = DictConfig(
        {'preprocessor': DictConfig(preprocessor), 'encoder': DictConfig(encoder), 'decoder': DictConfig(decoder)}
    )

    model_instance = EncDecCTCModel(cfg=modelConfig)
    return model_instance


@pytest.fixture()
def rnnt_model():
    preprocessor = {'cls': 'nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor', 'params': dict({})}

    # fmt: off
    labels = [' ', 'a', 'b', 'c', 'd', 'e', 'f',
              'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
              'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
              'x', 'y', 'z', "'",
              ]
    # fmt: on

    model_defaults = {'enc_hidden': 96, 'pred_hidden': 64}

    # Test case where Encoder (default) is not adapter compatible
    encoder = {
        'cls': 'nemo.collections.asr.modules.ConvASREncoder',
        'params': {
            'feat_in': 64,
            'activation': 'relu',
            'conv_mask': True,
            'jasper': [
                {
                    'filters': model_defaults['enc_hidden'],
                    'repeat': 1,
                    'kernel': [1],
                    'stride': [1],
                    'dilation': [1],
                    'dropout': 0.0,
                    'residual': False,
                    'separable': True,
                    'se': True,
                    'se_context_size': -1,
                }
            ],
        },
    }

    decoder = {
        '_target_': 'nemo.collections.asr.modules.RNNTDecoder',
        'prednet': {'pred_hidden': model_defaults['pred_hidden'], 'pred_rnn_layers': 1},
    }

    joint = {
        '_target_': 'nemo.collections.asr.modules.RNNTJoint',
        'jointnet': {'joint_hidden': 32, 'activation': 'relu'},
    }

    decoding = {'strategy': 'greedy_batch', 'greedy': {'max_symbols': 10}}

    loss = {'loss_name': 'default', 'warprnnt_numba_kwargs': {'fastemit_lambda': 0.001}}

    modelConfig = DictConfig(
        {
            'labels': ListConfig(labels),
            'preprocessor': DictConfig(preprocessor),
            'model_defaults': DictConfig(model_defaults),
            'encoder': DictConfig(encoder),
            'decoder': DictConfig(decoder),
            'joint': DictConfig(joint),
            'decoding': DictConfig(decoding),
            'loss': DictConfig(loss),
        }
    )

    model_instance = EncDecRNNTModel(cfg=modelConfig)
    return model_instance


def get_adapter_cfg(in_features=50, dim=100, norm_pos='pre', atype='linear', **kwargs):
    valid_types = ['linear', 'mha', 'relmha']
    if atype not in valid_types:
        raise ValueError(f"Invalid type. Valid types = {atype}")

    if atype == 'linear':
        cfg = adapter_modules.LinearAdapterConfig(in_features=in_features, dim=dim, norm_position=norm_pos)
    elif atype == 'mha':
        cfg = multi_head_attention_adapter_module.MultiHeadAttentionAdapterConfig(
            n_head=kwargs.get('n_head', 1), n_feat=in_features
        )
    elif atype == 'relmha':
        cfg = multi_head_attention_adapter_module.RelPositionMultiHeadAttentionAdapterConfig(
            n_head=kwargs.get('n_head', 1), n_feat=in_features
        )

    print(cfg._target_)

    cfg = OmegaConf.structured(cfg)
    return cfg


class TestASRAdapterMixin:
    @pytest.mark.unit
    def test_class_paths_are_correct(self):
        # Resolve all object names in module
        obj_keys = list(dir(adapter_utils))
        for key in obj_keys:
            if 'CLASSPATH' in key:
                classpath = getattr(adapter_utils, key)
                # This will raise import error if it fails
                _ = model_utils.import_class_by_path(classpath)

                # Try getting thmulti_head_attention_adapter_module.pye config of the class
                config_path = classpath + "Config"
                _ = model_utils.import_class_by_path(config_path)

    @pytest.mark.unit
    def test_asr_model_constructor(self, model):
        original_num_params = model.num_weights

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

    @pytest.mark.unit
    def test_asr_model_constructor_mha_adapter(self, model):
        with pytest.raises(ValueError):
            model.add_adapter(name='adapter_0', cfg=get_adapter_cfg(atype='mha'))

    @pytest.mark.unit
    def test_conformer_constructor_mha_adapter(self, conformer_ctc_adapter):
        original_num_params = conformer_ctc_adapter.num_weights

        conformer_ctc_adapter.add_adapter(name='adapter_0', cfg=get_adapter_cfg(atype='relmha'))
        new_num_params = conformer_ctc_adapter.num_weights
        assert new_num_params > original_num_params

    @pytest.mark.unit
    def test_squeezeformer_constructor_mha_adapter(self, squeezeformer_ctc_adapter):
        original_num_params = squeezeformer_ctc_adapter.num_weights

        squeezeformer_ctc_adapter.add_adapter(name='adapter_0', cfg=get_adapter_cfg(atype='relmha'))
        new_num_params = squeezeformer_ctc_adapter.num_weights
        assert new_num_params > original_num_params

    @pytest.mark.unit
    def test_asr_model_constructor_encoder_module(self, model):
        original_num_params = model.num_weights

        model.add_adapter(name='encoder:adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

    @pytest.mark.unit
    def test_asr_model_constructor_decoder_module(self, model):
        original_num_params = model.num_weights

        model.add_adapter(name='decoder:adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params
        assert model.decoder.is_adapter_available()
        assert model.decoder.get_enabled_adapters()[0] == 'adapter_0'

    @pytest.mark.unit
    def test_asr_model_constructor_joint_module_ctc_skip(self, model):
        original_num_params = model.num_weights

        # this step should exit without adding adapters and without errors
        model.add_adapter(name='joint:adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params == original_num_params

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    def test_asr_model_constructor_joint_module_rnnt(self, rnnt_model):
        original_num_params = rnnt_model.num_weights

        rnnt_model.add_adapter(name='joint:adapter_0', cfg=get_adapter_cfg())
        new_num_params = rnnt_model.num_weights
        assert new_num_params > original_num_params
        assert rnnt_model.joint.is_adapter_available()
        assert rnnt_model.joint.get_enabled_adapters()[0] == 'adapter_0'

    @pytest.mark.unit
    def test_asr_multiple_adapter(self, model):
        original_num_params = model.num_weights

        model.add_adapter(name='adapter_0', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

        original_num_params = new_num_params
        model.add_adapter(name='adapter_1', cfg=get_adapter_cfg())
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

    @pytest.mark.unit
    @pytest.mark.parametrize('name', ['adapter_0', 'encoder:adapter_0', 'decoder:adapter_0'])
    def test_asr_forward_linear_pre(self, model, name):
        model.eval()
        torch.random.manual_seed(0)
        input_signal = torch.randn(2, 512)
        input_signal_length = torch.tensor([512, 512], dtype=torch.int32)

        origial_output = model(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        model.add_adapter(name=name, cfg=get_adapter_cfg())
        new_output = model(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize('name', ['adapter_0', 'encoder:adapter_0', 'decoder:adapter_0'])
    def test_asr_forward_linear_post(self, model, name):
        model.eval()
        torch.random.manual_seed(0)
        input_signal = torch.randn(2, 512)
        input_signal_length = torch.tensor([512, 512], dtype=torch.int32)

        origial_output = model(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        model.add_adapter(name=name, cfg=get_adapter_cfg(norm_pos='post'))
        new_output = model(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize('name', ['adapter_0', 'encoder:adapter_0'])
    def test_conformer_forward_mha(self, conformer_ctc_adapter, name):
        conformer_ctc_adapter.eval()
        torch.random.manual_seed(0)
        input_signal = torch.randn(2, 512)
        input_signal_length = torch.tensor([512, 512], dtype=torch.int32)

        origial_output = conformer_ctc_adapter(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        conformer_ctc_adapter.add_adapter(name=name, cfg=get_adapter_cfg(in_features=128, atype='mha'))
        new_output = conformer_ctc_adapter(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize('name', ['adapter_0', 'encoder:adapter_0'])
    def test_squeezeformer_forward_mha(self, squeezeformer_ctc_adapter, name):
        squeezeformer_ctc_adapter.eval()
        torch.random.manual_seed(0)
        input_signal = torch.randn(2, 512)
        input_signal_length = torch.tensor([512, 512], dtype=torch.int32)

        origial_output = squeezeformer_ctc_adapter(input_signal=input_signal, input_signal_length=input_signal_length)[
            0
        ]

        squeezeformer_ctc_adapter.add_adapter(name=name, cfg=get_adapter_cfg(in_features=128, atype='mha'))
        new_output = squeezeformer_ctc_adapter(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize('name1', ['adapter_0', 'encoder:adapter_0', 'decoder:adapter_0'])
    @pytest.mark.parametrize('name2', ['adapter_1', 'encoder:adapter_1', 'decoder:adapter_1'])
    def test_asr_multi_adapter_forward(self, model, name1, name2):
        model.eval()
        torch.random.manual_seed(0)
        input_signal = torch.randn(2, 512)
        input_signal_length = torch.tensor([512, 512], dtype=torch.int32)

        origial_output = model(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        model.add_adapter(name=name1, cfg=get_adapter_cfg())
        model.add_adapter(name=name2, cfg=get_adapter_cfg())
        new_output = model(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        resolved_name1 = model.resolve_adapter_module_name_(name1)[-1]
        resolved_name2 = model.resolve_adapter_module_name_(name2)[-1]
        assert model.get_enabled_adapters() == [resolved_name1, resolved_name2]
        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.parametrize('name1', ['decoder:adapter_0', 'joint:adapter_0'])
    @pytest.mark.parametrize('name2', ['decoder:adapter_1', 'joint:adapter_1'])
    @pytest.mark.unit
    def test_asr_multi_adapter_forward(self, rnnt_model, name1, name2):
        rnnt_model.eval()
        torch.random.manual_seed(0)
        input_signal = torch.randn(2, 512)
        input_signal_length = torch.tensor([512, 512], dtype=torch.int32)

        origial_output = rnnt_model(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        rnnt_model.add_adapter(name=name1, cfg=get_adapter_cfg())
        rnnt_model.add_adapter(name=name2, cfg=get_adapter_cfg())
        new_output = rnnt_model(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        resolved_name1 = rnnt_model.resolve_adapter_module_name_(name1)[-1]
        resolved_name2 = rnnt_model.resolve_adapter_module_name_(name2)[-1]
        assert rnnt_model.get_enabled_adapters() == [resolved_name1, resolved_name2]
        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize('name1', ['adapter_0', 'encoder:adapter_0', 'decoder:adapter_0'])
    @pytest.mark.parametrize('name2', ['adapter_1', 'encoder:adapter_1', 'decoder:adapter_1'])
    def test_asr_multi_adapter_partial_forward(self, model, name1, name2):
        model.eval()
        torch.random.manual_seed(0)
        input_signal = torch.randn(2, 512)
        input_signal_length = torch.tensor([512, 512], dtype=torch.int32)

        origial_output = model(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        model.add_adapter(name=name1, cfg=get_adapter_cfg())
        model.add_adapter(name=name2, cfg=get_adapter_cfg())

        model.set_enabled_adapters(name=name1, enabled=False)
        new_output = model(input_signal=input_signal, input_signal_length=input_signal_length)[0]

        resolved_name2 = model.resolve_adapter_module_name_(name2)[-1]
        assert model.get_enabled_adapters() == [resolved_name2]
        assert torch.mean(torch.abs(origial_output - new_output)) < 1e-5

    @pytest.mark.unit
    @pytest.mark.parametrize('name', ['adapter_0', 'encoder:adapter_0', 'decoder:adapter_0'])
    def test_asr_forward_unfrozen_adapters(self, model, name):
        model.eval()
        original_num_params = model.num_weights

        dim = 10
        model.add_adapter(name=name, cfg=get_adapter_cfg(dim=dim))
        model.freeze()
        model.unfreeze_enabled_adapters()

        assert original_num_params == 5443

        original_params = 0
        adapter_params = 0
        for name, param in model.named_parameters():
            if 'adapter' not in name:
                assert param.requires_grad is False
                original_params += param.numel()
            else:
                assert param.requires_grad is True
                adapter_params += param.numel()

        for mname, module in model.named_modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                assert module.track_running_stats is False

        assert original_params > adapter_params

    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_constructor_pretrained(self):
        # Check to/from config_dict:
        cfg = ASRModel.from_pretrained('stt_en_citrinet_256', map_location='cpu', return_config=True)
        adapter_metadata = get_registered_adapter(cfg.encoder._target_)
        if adapter_metadata is not None:
            cfg.encoder._target_ = adapter_metadata.adapter_class_path
        model = ASRModel.from_pretrained('stt_en_citrinet_256', override_config_path=cfg)

        assert isinstance(model, AdapterModuleMixin)
        assert hasattr(model, 'encoder')
        assert isinstance(model.encoder, AdapterModuleMixin)

        model.add_adapter('adapter_0', cfg=get_adapter_cfg(in_features=cfg.encoder.jasper[0].filters, dim=5))
        assert model.is_adapter_available()

        model.freeze()
        model.unfreeze_enabled_adapters()
        assert model.num_weights < 1e5

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE, reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads()
    @pytest.mark.unit
    def test_constructor_pretrained_rnnt(self):
        # Check to/from config_dict:
        cfg = ASRModel.from_pretrained('stt_en_contextnet_256', map_location='cpu', return_config=True)
        adapter_metadata = get_registered_adapter(cfg.encoder._target_)
        if adapter_metadata is not None:
            cfg.encoder._target_ = adapter_metadata.adapter_class_path
        model = ASRModel.from_pretrained('stt_en_contextnet_256', override_config_path=cfg)

        assert isinstance(model, AdapterModuleMixin)
        assert hasattr(model, 'encoder')
        assert isinstance(model.encoder, AdapterModuleMixin)
        assert hasattr(model, 'decoder')
        assert isinstance(model.decoder, AdapterModuleMixin)
        assert hasattr(model, 'joint')
        assert isinstance(model.joint, AdapterModuleMixin)

        model.add_adapter('adapter_0', cfg=get_adapter_cfg(in_features=cfg.encoder.jasper[0].filters, dim=5))
        model.add_adapter('decoder:adapter_1', cfg=get_adapter_cfg(in_features=cfg.decoder.prednet.pred_hidden, dim=5))
        model.add_adapter('joint:adapter_2', cfg=get_adapter_cfg(in_features=cfg.joint.jointnet.joint_hidden, dim=5))
        assert model.is_adapter_available()

        model.freeze()
        model.unfreeze_enabled_adapters()
        assert model.num_weights < 1e5

    @pytest.mark.unit
    def test_asr_model_adapter_loss(self, model):
        original_num_params = model.num_weights
        x = torch.randn(2, 512)
        x_len = torch.tensor([256, 512], dtype=torch.int32)

        adapter_cfg = get_adapter_cfg()  # type: adapter_modules.LinearAdapterConfig
        adapter_cfg.adapter_strategy.l2_lambda = 0.01

        model.add_adapter(name='adapter_0', cfg=adapter_cfg)
        new_num_params = model.num_weights
        assert new_num_params > original_num_params

        model.train()  # set training mode to true

        with torch.no_grad():
            AccessMixin.reset_registry(model)
            AccessMixin.update_access_cfg({'save_encoder_tensors': False}, model.model_guid)
            _ = model(input_signal=x, input_signal_length=x_len)

            # extract losses
            auxiliary_losses = AccessMixin.get_module_registry(model)

            loss = list(auxiliary_losses.values())[0]
            assert 'adapter_loss' in loss
            assert loss['adapter_loss'][0] == torch.tensor(0.0)  # initially adapter is 0 init, no loss required.

            AccessMixin.reset_registry(model)
