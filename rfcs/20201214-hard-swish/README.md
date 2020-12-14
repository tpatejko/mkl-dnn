# Proposal to introduce hard-swish activation function

## Introduction
This document proposes implementation of the hard-swish activation function. The feature was requested by a team working on oneDNN integration in PaddlePaddle.
According to profile information obtained by the team, naive implementation available in PaddlePaddle could take 10% of the execution time of the MobileNetV3 model.

As the hard-swish function follows convolution in the customer's models, suggested method of implementation is as post-op.

Hardswish in PyTorch:
Link
https://pytorch.org/docs/stable/nn.functional.html?highlight=hardswish#torch.nn.functional.hardswish

Interface:
```
hardswish(input: torch.Tensor, inplace: bool = False) → torch.Tenso
```

Hardswish in PaddlePaddle:
Link:
https://www.paddlepaddle.org.cn/documentation/docs/api_cn/layers_cn/hard_swish_cn.html

Interface:
```
hard_swish(x, threshold=6.0, scale=6.0, offset=3.0, name=None)
```

Hardswish in OpenVino:
https://github.com/openvinotoolkit/openvino/blob/master/docs/ops/activation/HSwish_4.md

Interface:
```
<layer ... type="HSwish">
    <input>
	<port id="0">
        	<dim>256</dim>
		<dim>56</dim>
	</port>
    </input>
    <output>
	<port id="1">
	    <dim>256</dim>
	    <dim>56</dim>
	</port>
    </output>
</layer>
```
