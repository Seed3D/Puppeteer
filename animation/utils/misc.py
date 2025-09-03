#  Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from torch.optim.lr_scheduler import LambdaLR

def warmup_then_decay(optimizer, total_steps, warmup_steps, max_lr=1e-3, min_lr=1e-5, base_lr=1e-5):
    """
    Create a learning rate scheduler with warmup followed by decay.
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # warmup: min_lr -> max_lr
            progress = float(current_step) / float(max(1, warmup_steps))
            # LR(t) = min_lr + (max_lr - min_lr)*progress
            return (min_lr + (max_lr - min_lr)*progress) / base_lr
        else:
            # decay: warmup_steps -> total_steps
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            # LR(t) = max_lr + (min_lr - max_lr)*progress
            return (max_lr + (min_lr - max_lr)*progress) / base_lr

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler