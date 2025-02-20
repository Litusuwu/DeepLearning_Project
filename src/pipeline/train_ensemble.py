import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Average, Input

densenet_path = "experiments/individual_models/densenet/densenet_final.keras"
resnet_path = "experiments/individual_models/resnet/resnet_final.keras"
xception_path = "experiments/individual_models/xception/xception_final.keras"

model_densenet = load_model(densenet_path)
model_resnet = load_model(resnet_path)
model_xception = load_model(xception_path)

# freeze the models
model_densenet.trainable = False
model_resnet.trainable = False
model_xception.trainable = False

# assume the same input shape for all
input_shape = model_densenet.input_shape[1:]
inp = Input(shape=input_shape)

# Get each model’s prediction
pred1 = model_densenet(inp)
pred2 = model_resnet(inp)
pred3 = model_xception(inp)

# average the predictions
ensemble_output = Average()([pred1, pred2, pred3])
ensemble_model = Model(inputs=inp, outputs=ensemble_output)
ensemble_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# save the ensemble model
save_path = "experiments/individual_models/ensemble_model.keras"
ensemble_model.save(save_path)
print(f"✅ Ensemble model saved at {save_path}")
