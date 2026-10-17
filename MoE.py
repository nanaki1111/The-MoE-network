import scipy.io as scio
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Lambda, Activation, BatchNormalization, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from utils import yc_patch, yc_patch_inv


def build_expert(input_size, name):
    inp = Input(shape=(input_size,))
    x = Dense(128)(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(input_size)(x)
    return Model(inp, x, name=name)

# Gating 
def build_gating(input_size):
    inp = Input(shape=(input_size,))
    x = Dense(64, activation='relu')(inp)
    x = Dense(3)(x)
    weights = Activation('softmax')(x)
    return Model(inp, weights, name="Gate")

# Entropy Regularizer 
class EntropyRegularizer(Layer):
    def __init__(self, alpha=0.01, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.epsilon = epsilon

    def call(self, weights):
        entropy = -tf.reduce_sum(
            weights * tf.math.log(weights + self.epsilon),
            axis=1
        )

        entropy_loss = tf.reduce_mean(entropy)

        self.add_loss(-self.alpha * entropy_loss)

        return weights

# SSIM 
def compute_ssim(x, y):
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    mu_x = np.mean(x)
    mu_y = np.mean(y)

    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    L = np.max([x.max(), y.max()]) - np.min([x.min(), y.min()])
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    return ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2) + 1e-8)

# SNR 
def compute_snr(clean, noisy):
    noise = clean - noisy
    return 10*np.log10(np.sum(clean**2) / (np.sum(noise**2) + 1e-8))

#  Load data
clean = scio.loadmat(
    r'./data/clean_data.mat'
)['data']


noisy = scio.loadmat(
    r'./data/noisy_data.mat'
)['noisy_data']

print("shape:", noisy.shape)

# ---------------- Patch ------------------
w1, w2 = 48, 48
z1, z2 = 1, 1

n1, n2 = noisy.shape

x_train = yc_patch(noisy, w1, w2, z1, z2).astype('float32')

mean = np.mean(x_train)
std = np.std(x_train) + 1e-8
x_train = (x_train - mean) / std

input_size = x_train.shape[1]

#  Model 
expert_a = build_expert(input_size, "Expert_A")
expert_b = build_expert(input_size, "Expert_B")
expert_c = build_expert(input_size, "Expert_C")

gate = build_gating(input_size)

inp = Input(shape=(input_size,))

weights = gate(inp)

# entropy regularizer
weights = EntropyRegularizer(alpha=0.01)(weights)

out_a = expert_a(inp)
out_b = expert_b(inp)
out_c = expert_c(inp)

out = Lambda(lambda x: (
    x[0][:, 0:1]*x[1] +
    x[0][:, 1:2]*x[2] +
    x[0][:, 2:3]*x[3]
))([weights, out_a, out_b, out_c])

model = Model(inp, out)

model.compile(
    optimizer=Adam(1e-3),
    loss='mse'
)

gate_model = Model(inp, weights)

# Train 
model.fit(
    x_train,
    x_train,
    epochs=200,
    batch_size=128,
    shuffle=True,
    verbose=2
)


# Inference 
patch_eval = yc_patch(noisy, w1, w2, z1, z2).astype('float32')
patch_eval = (patch_eval - mean) / std

den_patch = model.predict(patch_eval, verbose=0)

denoised = yc_patch_inv(den_patch.T, n1, n2, w1, w2, z1, z2)
denoised = denoised * std + mean


clean_patch = yc_patch(clean, w1, w2, z1, z2).astype('float32')
clean_patch = (clean_patch - mean) / std

clean_recon = yc_patch_inv(clean_patch.T, n1, n2, w1, w2, z1, z2)
clean_recon = clean_recon * std + mean


print("\n===== Metrics =====")
print("SNR noisy   :", compute_snr(clean_recon, noisy))
print("SNR denoise :", compute_snr(clean_recon, denoised))
print("SSIM noisy  :", compute_ssim(clean_recon, noisy))
print("SSIM denoise:", compute_ssim(clean_recon, denoised))

# ---------------- Gating ------------------
g = gate_model.predict(patch_eval, verbose=0)

print("\nExpert weights:")
print(np.mean(g, axis=0))

# Plot 
c = 1000

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(noisy, cmap='seismic', vmin=-c, vmax=c)
plt.title("Noisy")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(denoised, cmap='seismic', vmin=-c, vmax=c)
plt.title("Denoised")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(noisy - denoised, cmap='seismic', vmin=-c, vmax=c)
plt.title("Residual")
plt.axis('off')

plt.show()

# scio.savemat(r'./data/denoise.mat', {'denoise': denoised})