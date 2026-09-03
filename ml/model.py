import torch
from torch import nn

# nn.Module:
# It gives PyTorch the machinery needed to:

# - track the network's weights;
# - move it to CPU/GPU;
# - save/load it;
# - calculate gradients;
# - switch between training/evaluation modes.

class ArabicLetterCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # nn.Sequential:
        # Instead of manually doing:
            # x = self.conv1(x)
            # x = self.relu1(x)
            # x = self.pool1(x)
            # x = self.conv2(x)
            # ...

        # we can define:
            # self.features = nn.Sequential(...)    
            # Pass x through these layers in order.


        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input: 1 × 64 × 121
                out_channels=16,  # output: 16 feature maps
                kernel_size=3,
                padding=1,
            ),
            # ReLU:
                # negative number → 0
                # positive number → keep it
            # i.e ReLU(x) = max(0, x)
            nn.ReLU(),
            nn.MaxPool2d(2),  # After pooling: 16 × 32 × 60

            # second convolution examines those 16 learned feature maps
            # and creates 32 more sophisticated one
            nn.Conv2d(
                in_channels=16, 
                out_channels=32,
                kernel_size=3,
                padding=1
                ),
            nn.ReLU(),
            nn.MaxPool2d(2),  # After pooling 32 x 16 x 30
        )

        # AdaptiveAvgPool2d((4, 4)):
            # After the convolutions and MaxPool layers, each recording is:
            # represented approximately as:
            # [32, 16, 30]
                # 32 learned feature maps
                # 16 positions along the frequency axis
                # 30 positions along the time axis

            # Our OLD model used:
                # AdaptiveAvgPool2d((1, 1))

            # which changed:
                # 32 × 16 × 30
                #    ↓
                # 32 × 1 × 1
            
            # This averaged every feature map down to just one number.
            # That meant the network knew roughly:
                # "How strongly was this feature present?"

            # but lost most information about:
                # WHERE in frequency it appeared
                # WHERE in time it appeared

            # That is a problem for speech because the location of acoustic
            # features in frequency and time can help distinguish sounds.

            # Our NEW model therefore uses:
                # AdaptiveAvgPool2d((4, 4))

            # This changes:
                # 32 × 16 × 30
                #       ↓
                # 32 × 4 × 4

            # So instead of one number per feature map,
            # we preserve a coarse 4 × 4 representation of each feature map.

            # This gives:
                # 32 feature maps
                # × 4 frequency regions
                # × 4 time regions
                # = 512 numbers

        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        # CLASSIFIER:
            # After AdaptiveAvgPool2d((4, 4)), each recording has shape:
                # [32, 4, 4]

            # Then in forward():
                # torch.flatten(x, start_dim=1)

            # changes:
                # [batch, 32, 4, 4]
                #         ↓
                # [batch, 512]

            # because:
                # 32 × 4 × 4 = 512

            # So every recording is now represented by 512 learned features.

            # The classifier then does:

                # 512 features
                #      ↓
                # Linear layer
                #      ↓
                # 64 hidden features
                #      ↓
                # ReLU
                #      ↓
                # Linear layer
                #      ↓
                # 6 output logits

            # First Linear layer:
                # nn.Linear(512, 64)

            # This lets the model learn useful combinations of the
            # 512 convolutional features.

            # For example, conceptually, it could learn that:
                # a certain high-frequency feature
                #        +
                # a certain short-duration feature
                #        +
                # their approximate positions
                #        ↓
                # together provide evidence for a particular sound

            # We do NOT manually tell the network these combinations.
            # Training adjusts the weights so that useful combinations
            # are learned automatically.

            # ReLU then adds non-linearity:
                # negative value → 0
                # positive value → keep it

            # The final Linear layer:
                # nn.Linear(64, 6)

            # converts those 64 learned hidden features into 6 logits:
                # 0 → qaf
                # 1 → kaf
                # 2 → ta
                # 3 → taa_emphatic
                # 4 → sin
                # 5 → sad

            # The logits are NOT probabilities.
            # During training, CrossEntropyLoss works directly with them.

        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=32 * 4 * 4,  # 512
                out_features=64,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=64,
                out_features=num_classes,  # 6
            ),
        )


    def forward(self, x):
        x = self.features(x)

        x = self.global_pool(x)

        x = torch.flatten(x, start_dim=1)

        x = self.classifier(x)

        return x
