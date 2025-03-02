import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

teacher_model = load_model("experiments/individual_models/ensemble_model.keras")
teacher_model.trainable = False

input_shape = (224, 224, 3)
inp = Input(shape=input_shape)
base_student = MobileNetV3Small(weights='imagenet', include_top=False, input_tensor=inp)
x = GlobalAveragePooling2D()(base_student.output)
student_output = Dense(1, activation='sigmoid')(x)
student_model = Model(inputs=inp, outputs=student_output)

student_model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])


alpha = 0.5 

@tf.function
def train_step(x, y):
    teacher_preds = teacher_model(x, training=False)
    with tf.GradientTape() as tape:
        student_preds = student_model(x, training=True)
        loss_student = tf.keras.losses.binary_crossentropy(y, student_preds)
        loss_distill = tf.keras.losses.mean_squared_error(teacher_preds, student_preds)
        loss = alpha * loss_student + (1 - alpha) * loss_distill
    gradients = tape.gradient(loss, student_model.trainable_variables)
    student_model.optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))
    return loss

train_datagen = ImageDataGenerator(rescale=1./255)
train_dir = "data/processed/Training"
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=input_shape[:2], batch_size=32, class_mode='binary'
)

epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for batch_x, batch_y in train_generator:
        loss = train_step(batch_x, batch_y)
    print(f"Loss: {loss.numpy().mean():.4f}")
    if train_generator.batch_index == 0:
        break

save_path = "experiments/individual_models/distilled_student.keras"
student_model.save(save_path)
print(f"✅ Distilled student model saved at {save_path}")
