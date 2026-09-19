import copy
from pathlib import Path

import torch
from torch import nn

from ml.data_loaders import create_data_loaders
from ml.labels import CLASSES
from ml.model import ArabicLetterCNN

MAX_EPOCHS = 200
LEARNING_RATE = 0.001
USE_AUGMENTATION = True
EARLY_STOPPING_PATIENCE = 30
MODEL_FILE = Path(
    "models/latest/arabic_letter_cnn.pt"
)


def calculate_accuracy(outputs, labels):
    predictions = outputs.argmax(dim=1) 

    correct = (predictions == labels).sum().item()

    return correct


def main():
    # RANDOM SEED:
        # Neural-network training contains several random processes:
            # - the model's initial weights
            # - the order of shuffled training batches
            # - our random data augmentation
        
        # Setting the PyTorch random seed makes those random choices
        # reproducible.
        
        # This does NOT mean that everything stops being random.
        # It means PyTorch generates the same sequence of random
        # values each time we start this program with the same seed.
        
        # This is useful experimentally because if we run the exact
        # same experiment twice, we should get much more comparable results.

    torch.manual_seed(42)

    train_loader, validation_loader, _ = create_data_loaders(
        batch_size=4,
        augment_training=USE_AUGMENTATION
    )

    model = ArabicLetterCNN(num_classes=len(CLASSES))

    # CrossEntropyLoss:
        # How wrong was the network

        # Our network outputs six logits:
            # qaf           0.2
            # kaf          -0.4
            # ta            0.8
            # taa_emphatic  1.4
            # sin          -0.1
            # sad           0.3

        # suppose the actual answer was: label = 2 i.e تَ
        # But the network's largest score was: class 3 → طَ

        # Cross-entropy gives us a number representing how bad that prediction was.
        # CrossEntropyLoss is specifically designed to take unnormalized logits

        # Initially you might see something around: loss ≈ 1.8
        # probability each ≈ 1/6 -> -ln(1/6) ≈ 1.79 (ln = natural log)

        # As it learns, hopefully we'll see:
            # 1.79
            # 1.62
            # 1.39
            # 1.07
            # 0.73
            # 0.41
            # ...

            # Lower is better.

    loss_function = nn.CrossEntropyLoss() 

    # Adam:
        # the algorithm responsible for changing the weights
        # based on the gradients backpropagation calculated.

    # lr=0.001:
        # This is the learning rate.
        # it is roughly controlling the size of each adjustment.

        # too large: model overshoots

        # too small: tiny tiny tiny changes -> takes forever

        # 0.001 is a sensible starting point for Adam
        # it isn't a golden universal value.

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Best validation loss seen so far.

    # Start at infinity because the first real validation loss
    # will always be smaller.
    best_validation_loss = float("inf")

    # We will store a copy of the best model weights here.
    best_model_state = None

    best_epoch = None

    # EARLY STOPPING:
    # Count how many consecutive epochs have passed
    # without improving the best validation loss.
    
    # Whenever validation improves:
        # reset this back to 0
    
    # Whenever validation does NOT improve:
        # add 1
    
    # Once it reaches EARLY_STOPPING_PATIENCE:
        # stop training

    epochs_without_improvement = 0

    print(f"Data augmentation: {USE_AUGMENTATION}")
    print("Training Arabic Letter CNN")
    print("--------------------------")
    print(f"Maximum epochs: {MAX_EPOCHS}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()
    for epoch in range(1, MAX_EPOCHS + 1):
        # Training
            
        model.train()

        training_loss = 0.0
        training_correct = 0
        training_samples = 0

        for spectrograms, labels in train_loader:

            # NN training for one batch is simply:

                # 1. FORWARD PASS
                    # spectrogram 
                    #    ↓
                    #   CNN
                    #    ↓
                    # prediction

                # 2. LOSS
                    # prediction
                    #     ↕
                    # correct answer

                    # i.e How wrong were the predictions?

                # 3. BACKPROPAGATION
                    #          loss
                    #           ↓
                    # calculate how each trainable weight
                    # contributed to that error

                # 4. OPTIMISATION
                    # use those gradients to slightly
                    # change the weights in a direction
                    # intended to reduce the loss
                
            # Then repeat for the next batch.


            # step 1: FORWARD PASS
            # Ask the model for predictions
            outputs = model(spectrograms)

            # Step 2: LOSS
            # Measure how wrong those predictions are
            loss = loss_function(outputs, labels)

            # Clear gradients left over from the previous batch
            # PyTorch accumulates gradients by default, so without this:
                # Batch 1 gradients
                #      +
                # Batch 2 gradients
                #      +
                # Batch 3 gradients
                #     
            # would keep accumulating.
            optimizer.zero_grad()

            # Step 3: BACKPROPAGATION
            # Calculate the gradient of the loss with respect
            # to each trainable weight in the network.
            loss.backward()

            # Step 4: OPTIMISATION
            # Update the neural-network weights
            # Adam uses those gradients to update the weights.
            optimizer.step()

            training_loss += loss.item() * labels.size(0)

            training_correct += calculate_accuracy(
                outputs,
                labels
            )

            training_samples += labels.size(0)

        training_loss /= training_samples

        training_accuracy = (
            training_correct / training_samples
        )

        # validation

        model.eval()

        validation_loss = 0.0
        validation_correct = 0
        validation_samples = 0

        with torch.no_grad():
            for spectrograms, labels in validation_loader:

                outputs = model(spectrograms)

                loss = loss_function(outputs, labels)

                validation_loss += (
                    loss.item() * labels.size(0)
                )

                validation_correct += calculate_accuracy(outputs, labels)

                validation_samples += labels.size(0)

        validation_loss /= validation_samples

        validation_accuracy = (
            validation_correct / validation_samples
        )

        print(
            f"Epoch {epoch:02}/{MAX_EPOCHS} | "
            f"Train loss: {training_loss:.4f} | "
            f"Train accuracy: {training_accuracy:.1%} | "
            f"Validation loss: {validation_loss:.4f} | "
            f"Validation accuracy: {validation_accuracy:.1%}"
        )


        # BEST MODEL CHECKPOINT + EARLY STOPPING

        # If this epoch has a lower validation loss than every
        # previous epoch, it is our new best model.

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss
            best_epoch = epoch

            # Take a frozen copy of the weights from THIS epoch.
            
            # deepcopy is necessary because training will continue
            # modifying the model's actual weights afterwards.
            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            # Because validation improved, early stopping gets
            # a fresh start.
            epochs_without_improvement = 0

            print(
                f"  -> New best validation loss "
                f"({best_validation_loss:.4f})"
            )
        else:
            # Validation did not beat our best result.
            
            # Count this as one epoch without improvement.
            epochs_without_improvement += 1

            print(
                f"  -> No validation improvement "
                f"({epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE})"
            )

            # If validation has failed to improve for enough
            # consecutive epochs, stop training.
            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):
                print()
                print(
                    "Early stopping triggered: "
                    f"validation loss has not improved for "
                    f"{EARLY_STOPPING_PATIENCE} consecutive epochs."
                )

                break

    # Save best model

    if best_model_state is None:
        raise RuntimeError("No model checkpoint was created.")

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        best_model_state,
        MODEL_FILE,
    )

    print()
    print("Training finished.")
    print(f"Training stopped after epoch: {epoch}")
    print(f"Best epoch: {best_epoch}")
    print(
        f"Best validation loss: "
        f"{best_validation_loss:.4f}"
    )
    print(f"Best model saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()




