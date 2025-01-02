Models
======

End-to-End ASR models are typically of encoder-decoder style, where the encoder does acoustic 
modeling i.e., converting speech wavform into features, and the decoder converts those features into 
text. Encoder contains the bulk of trainable parameters and is usually the focus of SSL in ASR. 
Thus, any architecture that can be used as encoder in ASR models can be pre-trained using SSL. For an 
overview of model architectures that are currently supported in NeMo's ASR's collection, refer 
to `ASR Models <../models.html>`__. Note that SSL also uses encoder-decoder style of models. During 
down-stream fine-tuning, the encoder is retained where as the decoder (used during SSL) is replaced 
with down-stream task specific module. Refer to `checkpoints <./results.html>`__ to see how this is 
accomplished in NeMo. 
