import random

from torch.utils.data import DataLoader, Subset

from ml.dataset import CLASSES, ArabicLetterDataset

RANDOM_SEED = 42

TRAIN_PER_CLASS = 10
VALIDATION_PER_CLASS = 3
TEST_PER_CLASS = 2


def create_data_loaders(batch_size=4, augment_training=True):
    # We initially create a non-augmented dataset so that we can
    # determine the file indices belonging to each split.
    #
    # All ArabicLetterDataset instances discover files in the same
    # sorted order, so index 20 refers to the same WAV file in each one.
    split_dataset = ArabicLetterDataset(
        augment=False
    )

    # TRAINING dataset:
        # Only the training source is allowed to use augmentation.
        
        # augment_training=True:
        #     training recordings receive random transformations
        
        # augment_training=False:
        #     training recordings are used exactly as recorded
    
        # This switch lets us run a controlled experiment where
        # everything remains identical except augmentation.
    training_source = ArabicLetterDataset(
        augment=augment_training
    )

    # VALIDATION + TEST dataset:
        # augmentation disabled
    evaluation_source = ArabicLetterDataset(
        augment=False
    )

    train_indices = []
    validation_indices = []
    test_indices = []

    # We use our own Random object with a fixed seed.

    # Therefore the files assigned to:
        # training
        # validation
        # test

    # remain the same every time we run the program.
    rng = random.Random(RANDOM_SEED)

    for class_index in range(len(CLASSES)):
        # Find every dataset index belonging to this class

        # for example:
        # qaf might correspond to indices 0-14
        # kaf to 15-29, etc.

        class_indices = [
            index
            for index, (_, label) in enumerate(split_dataset.samples)
            if label == class_index
        ]

        rng.shuffle(class_indices)

        # First TRAIN_PER_CLASS recordings -> training

        train_indices.extend(class_indices[:TRAIN_PER_CLASS])

        # Next VALIDATION_PER_CLASS recordings -> validation

        validation_indices.extend(
            class_indices[TRAIN_PER_CLASS : TRAIN_PER_CLASS + VALIDATION_PER_CLASS]
        )

        # Everything remaining -> test

        test_indices.extend(class_indices[TRAIN_PER_CLASS + VALIDATION_PER_CLASS :])

    train_dataset = Subset(training_source, train_indices)
    validation_dataset = Subset(evaluation_source, validation_indices)
    test_dataset = Subset(evaluation_source, test_indices)

    # batch_size:
    # With batch_size=4:

    # 4 spectrograms
    #       ↓
    # make predictions
    #       ↓
    # calculate how wrong they were
    #       ↓
    # update model

    # Then the next batch is processed.

    # One complete pass through all training samples is called an epoch.
    # 1 epoch = CNN has seen all 24 training samples once
    # 50 epochs -> the network has gone through the training set 50 times, adjusting itself gradually.

    # The number of batches depends on:
        # number of training samples
        #             ÷
        #        batch size


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader
