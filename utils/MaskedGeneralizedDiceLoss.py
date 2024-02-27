# Adapted from MONAI Masked Dice Loss: https://docs.monai.io/en/stable/_modules/monai/losses/dice.html#MaskedDiceLoss

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
from monai.losses.dice import GeneralizedDiceLoss
from monai.losses.spatial_mask import MaskedLoss


class MaskedGeneralizedDiceLoss(GeneralizedDiceLoss):
    '''
    Masked Generalized Dice Loss
    '''

    """
    Add an additional `masking` process before `DiceLoss`, accept a binary mask ([0, 1]) indicating a region,
    `input` and `target` will be masked by the region: region with mask `1` will keep the original value,
    region with `0` mask will be converted to `0`. Then feed `input` and `target` to normal `DiceLoss` computation.
    This has the effect of ensuring only the masked region contributes to the loss computation and
    hence gradient calculation.

    Adapted from monai.losses.dice source code.

    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Args follow :py:class:`monai.losses.GeneralizedDiceLoss`.
        """
        super().__init__(*args, **kwargs)
        self.spatial_weighted = MaskedLoss(loss=super().forward)

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            mask: the shape should B1H[WD] or 11H[WD].
        """
        return self.spatial_weighted(input=input, target=target, mask=mask)
