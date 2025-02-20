import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Average, Input


def create_ensemble_model(model_paths):
    """
    Create an ensemble model that can be saved as a .keras file.
    Using a different approach to avoid naming conflicts.
    """
    # Load base models
    print("Loading models...")
    base_models = []
    for path in model_paths:
        print(f"Loading model from {path}")
        model = load_model(path)
        model.trainable = False
        base_models.append(model)

    # Get input shape from first model
    input_shape = base_models[0].input_shape[1:]

    # Create new models with separate inputs to avoid naming conflicts
    models = []
    inputs = []
    outputs = []

    for idx, base_model in enumerate(base_models):
        # Create new input layer for each model
        inp = Input(shape=input_shape, name=f'input_model_{idx}')
        inputs.append(inp)

        # Get the output for this input
        out = base_model(inp)
        outputs.append(out)

    # Average the outputs
    if len(outputs) > 1:
        ensemble_output = Average(name='ensemble_average')(outputs)
    else:
        ensemble_output = outputs[0]

    # Create the ensemble model
    ensemble_model = Model(
        inputs=inputs[0],  # Use only first input since all inputs will be the same
        outputs=ensemble_output,
        name='ensemble_model'
    )

    # Compile the model
    ensemble_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return ensemble_model


def main():
    # Define paths to your saved models
    model_paths = [
        "experiments/individual_models/densenet/densenet_final.keras",
        "experiments/individual_models/resnet/resnet_final.keras",
        "experiments/individual_models/xception/xception_final.keras"
    ]

    # Create and compile ensemble model
    print("Creating ensemble model...")
    ensemble_model = create_ensemble_model(model_paths)

    # Save the ensemble model
    save_path = "experiments/individual_models/ensemble_model.keras"
    ensemble_model.save(save_path)
    print(f"✅ Ensemble model saved at {save_path}")

    return ensemble_model


if __name__ == "__main__":
    ensemble_model = main()