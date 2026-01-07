import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

MODEL_VERSION = "v1"

def load_or_create_model():
    """Charge le modèle ou en crée un nouveau si inexistant"""
    model_path = "models/mnist_cnn.h5"

    if os.path.exists(model_path):
        print(f"✅ Chargement du modèle depuis {model_path}")
        model = load_model(model_path)

        # Tester le modèle sur quelques exemples
        print("🧪 Test du modèle chargé...")
        test_random_images(model)

        return model
    else:
        print(f"⚠️ Modèle non trouvé à {model_path}. Création d'un modèle entraîné.")
        return create_and_train_model()

def create_and_train_model():
    """Crée et entraîne un nouveau modèle MNIST"""
    print("📊 Chargement des données MNIST...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Prétraitement
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    print("🔨 Construction du modèle CNN...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("🚀 Entraînement du modèle (5 époques)...")
    model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    # Évaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"✅ Modèle entraîné avec succès!")
    print(f"   Précision sur test: {test_acc:.4f}")

    # Sauvegarde
    os.makedirs("models", exist_ok=True)
    model.save("models/mnist_cnn.h5")
    print(f"💾 Modèle sauvegardé dans models/mnist_cnn.h5")

    return model

def test_random_images(model):
    """Teste le modèle sur des images aléatoires"""
    # Images aléatoires
    random_imgs = np.random.rand(3, 28, 28, 1)

    print("   Tests sur images aléatoires:")
    for i, img in enumerate(random_imgs):
        pred = model.predict(img.reshape(1, 28, 28, 1), verbose=0)[0]
        pred_class = np.argmax(pred)
        confidence = np.max(pred)

        # Vérifier si le modèle prédit toujours la même chose
        if i == 0:
            first_pred = pred_class

        print(f"     Image {i+1}: classe={pred_class}, confiance={confidence:.2f}")

        # Afficher la distribution pour la première image
        if i == 0:
            print(f"       Distribution: {np.round(pred, 3)}")

    # Vérifier la diversité des prédictions
    print(f"   Diversité des prédictions: {'✅ OK' if len(set([np.argmax(model.predict(img.reshape(1,28,28,1), verbose=0)[0]) for img in random_imgs])) > 1 else '⚠️ TOUJOURS LA MÊME'}")


def predict_digit(img_array: np.ndarray):
    """
    Prédit le chiffre à partir d'un tableau numpy 28x28

    Args:
        img_array: Tableau numpy de shape (28, 28) avec valeurs 0-255

    Returns:
        tuple: (chiffre_prédit, confiance)
    """
    # Prétraitement
    img = img_array.astype('float32') / 255.0

    # Vérifier si l'image est inversée (fond blanc / chiffre noir)
    # Dans MNIST, le fond est noir (0) et le chiffre est blanc (1)
    if np.mean(img) > 0.5:
        print("🔁 Inversion des couleurs (fond blanc → fond noir)")
        img = 1.0 - img

    # Reshape pour le modèle
    img = img.reshape(1, 28, 28, 1)

    # Prédiction
    probs = model.predict(img, verbose=0)[0]
    predicted_class = int(np.argmax(probs))
    confidence = float(np.max(probs))

    # Debug
    print(f"🔍 Prédiction:")
    print(f"   Classe: {predicted_class}")
    print(f"   Confiance: {confidence:.2f}")

    # Si la confiance est faible ou toujours la même classe, afficher plus d'infos
    if confidence < 0.5:
        print(f"   ⚠️ Confiance faible!")
        top_3 = np.argsort(probs)[-3:][::-1]
        for i, cls in enumerate(top_3):
            print(f"   Top-{i+1}: classe {cls} ({probs[cls]:.3f})")

    return predicted_class, confidence


# Charger le modèle au démarrage
print("=" * 50)
print("🤖 Initialisation du modèle MNIST...")
model = load_or_create_model()
print(f"✅ Modèle prêt! Version: {MODEL_VERSION}")
print("=" * 50)
