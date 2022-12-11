import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, RNN, GRUCell, Input
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns


# Set you cpu or gpu device
device = tf.config.experimental.list_physical_devices('GPU')
if device:
    tf.config.experimental.set_memory_growth(device[0], True)

# Training parameters
sequence_len = 24
num_sequence = 6
batch_size = 128
num_hidden = 24
num_layers = 3
train_steps = 100000
gamma = 1
num_tol = 1e-6

# Optimization parameters
mse = MeanSquaredError()
bce = BinaryCrossentropy()
autoencoder_optimizer = Adam()
supervisor_optimizer = Adam()
generator_optimizer = Adam()
discriminator_optimizer = Adam()
embedding_optimizer = Adam()


# Load data and create directories
results_path = Path('time_gan')
if not results_path.exists():
    results_path.mkdir()

experiment = 0

log_dir = results_path / f'experiment_{experiment:02}'
if not log_dir.exists():
    log_dir.mkdir(parents=True)

hdf_store = results_path / 'TimeSeriesGAN.h5'

# Filtering data
stock_labels = ['XOM', 'NDAQ', 'AAPL', 'FB', 'GOOGL', 'AMZN']

df = (pd.read_hdf('assets.h5', 'quandl/wiki/prices').adj_close.unstack('ticker').loc['2000':, stock_labels].dropna())
df.to_hdf(hdf_store, 'data/real')
df = pd.read_hdf(hdf_store, 'data/real')

# Scalin data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df).astype(np.float32)

data = []
for i in range(len(df) - sequence_len):
    data.append(data_scaled[i:i + sequence_len])

number_windows = len(data)

series_real = (tf.data.Dataset.from_tensor_slices(data).shuffle(buffer_size=number_windows).batch(batch_size))
series_real_iter = iter(series_real.repeat())

# Create a random series
def make_random_data():
    while True:
        yield np.random.uniform(low=0, high=1, size=(sequence_len, num_sequence))

series_random = iter(tf.data.Dataset.from_generator(make_random_data, output_types=tf.float32).batch(batch_size).repeat())

# Create the recurrent nn
writer = tf.summary.create_file_writer(log_dir.as_posix())

X = Input(shape=[sequence_len, num_sequence], name='RealData')
Z = Input(shape=[sequence_len, num_sequence], name='RandomData')

def make_rnn(n_layers, hidden_units, output_units, name):
    return Sequential([GRU(units=hidden_units, return_sequences=True,name=f'GRU_{i + 1}') for i in range(n_layers)]+[Dense(units=output_units, activation='sigmoid',name='OUT')], name=name)

embedder = make_rnn(n_layers=3, hidden_units=num_hidden, output_units=num_hidden, name='Embedder')
recovery = make_rnn(n_layers=3, hidden_units=num_hidden, output_units=num_sequence, name='Recovery')
generator = make_rnn(n_layers=3, hidden_units=num_hidden, output_units=num_hidden, name='Generator')
discriminator = make_rnn(n_layers=3, hidden_units=num_hidden, output_units=1, name='Discriminator')
supervisor = make_rnn(n_layers=2, hidden_units=num_hidden, output_units=num_hidden, name='Supervisor')

# Create  and train the autoencoder
autoencoder = Model(inputs=X, outputs=recovery(embedder(X)), name='Autoencoder')
@tf.function
def train_autoencoder(x):
    with tf.GradientTape() as tape:
        x_til = autoencoder(x)
        loss_embedding_tstart = mse(x, x_til)
        e_loss_0 = 10 * tf.sqrt(loss_embedding_tstart)

    variable_list = embedder.trainable_variables + recovery.trainable_variables
    grads = tape.gradient(e_loss_0, variable_list)
    autoencoder_optimizer.apply_gradients(zip(grads, variable_list))
    return tf.sqrt(loss_embedding_tstart)

for step in tqdm(range(train_steps)):
    X_ = next(series_real_iter)
    e_loss_step_start = train_autoencoder(X_)
    with writer.as_default():
        tf.summary.scalar('Loss Autoencoder Init', e_loss_step_start, step=step)

# Train the supervisor
@tf.function
def train_supervisor(x):
    with tf.GradientTape() as tape:
        h = embedder(x)
        supervised_hhat = supervisor(h)
        g_loss_s = mse(h[:, 1:, :], supervised_hhat[:, :-1, :])

    variable_list = supervisor.trainable_variables
    grads = tape.gradient(g_loss_s, variable_list)
    supervisor_optimizer.apply_gradients(zip(grads, variable_list))
    return g_loss_s

for step in tqdm(range(train_steps)):
    X_ = next(series_real_iter)
    g_loss_s_step = train_supervisor(X_)
    with writer.as_default():
        tf.summary.scalar('Loss Generator Supervised Init', g_loss_s_step, step=step)

# Connect the adversarial training
Y_fake = discriminator(supervisor(generator(Z)))
adversarial_supervised = Model(inputs=Z, outputs=Y_fake, name='AdversarialNetSupervised')

Y_fake_e = discriminator(generator(Z))
adversarial_emb = Model(inputs=Z, outputs=Y_fake_e, name='AdversarialNet')

X_hat = recovery(supervisor(generator(Z)))
synthetic_data = Model(inputs=Z, outputs=X_hat,name='SyntheticData')

# Obtain discriminator loss
def get_generator_loss(y_true, y_pred):
    y_true_mean, y_true_var = tf.nn.moments(x=y_true, axes=[0])
    y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred, axes=[0])
    g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
    g_loss_var = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + num_tol) - tf.sqrt(y_pred_var + num_tol)))
    return g_loss_mean + g_loss_var

Y_real = discriminator(embedder(X))
discriminator_model = Model(inputs=X, outputs=Y_real, name='DiscriminatorReal')

# Train the generator
@tf.function
def train_generator(x, z):
    with tf.GradientTape() as tape:
        y_fake = adversarial_supervised(z)
        generator_loss_unsupervised = bce(y_true=tf.ones_like(y_fake), y_pred=y_fake)
        y_fake_e = adversarial_emb(z)
        generator_loss_unsupervised_e = bce(y_true=tf.ones_like(y_fake_e), y_pred=y_fake_e)
        h = embedder(x)
        supervised_hhat = supervisor(h)
        generator_loss_supervised = mse(h[:, 1:, :], supervised_hhat[:, 1:, :])

        x_hat = synthetic_data(z)
        generator_moment_loss = get_generator_loss(x, x_hat)

        generator_loss = (generator_loss_unsupervised + generator_loss_unsupervised_e + 100 * tf.sqrt(generator_loss_supervised) + 100 * generator_moment_loss)

    variable_list = generator.trainable_variables + supervisor.trainable_variables
    grads = tape.gradient(generator_loss, variable_list)
    generator_optimizer.apply_gradientsients(zip(grads, variable_list))
    return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss

# Train embedding
@tf.function
def train_embedder(x):
    with tf.GradientTape() as tape:
        h = embedder(x)
        supervised_hhat = supervisor(h)
        generator_loss_supervised = mse(h[:, 1:, :], supervised_hhat[:, 1:, :])

        x_til = autoencoder(x)
        loss_embedding_tstart = mse(x, x_til)
        e_loss = 10 * tf.sqrt(loss_embedding_tstart) + 0.1 * generator_loss_supervised

    variable_list = embedder.trainable_variables + recovery.trainable_variables
    grads = tape.gradient(e_loss, variable_list)
    embedding_optimizer.apply_gradientsients(zip(grads, variable_list))
    return tf.sqrt(loss_embedding_tstart)

# Train discriminator
@tf.function
def get_discriminator_loss(x, z):
    y_real = discriminator_model(x)
    discriminator_loss_real = bce(y_true=tf.ones_like(y_real), y_pred=y_real)

    y_fake = adversarial_supervised(z)
    discriminator_loss_fake = bce(y_true=tf.zeros_like(y_fake), y_pred=y_fake)

    y_fake_e = adversarial_emb(z)
    discriminator_loss_fake_e = bce(y_true=tf.zeros_like(y_fake_e), y_pred=y_fake_e)
    return (discriminator_loss_real + discriminator_loss_fake + gamma * discriminator_loss_fake_e)

@tf.function
def train_discriminator(x, z):
    with tf.GradientTape() as tape:
        discriminator_loss = get_discriminator_loss(x, z)

    variable_list = discriminator.trainable_variables
    grads = tape.gradient(discriminator_loss, variable_list)
    discriminator_optimizer.apply_gradientsients(zip(grads, variable_list))
    return discriminator_loss

# Finally train the whole thing!
step_g_loss_u = g_loss_s_step = step_g_loss_v = e_loss_step_start = step_d_loss = 0
for step in range(train_steps):
    for kk in range(2):
        X_ = next(series_real_iter)
        Z_ = next(series_random)

        step_g_loss_u, g_loss_s_step, step_g_loss_v = train_generator(X_, Z_)
        e_loss_step_start = train_embedder(X_)

    X_ = next(series_real_iter)
    Z_ = next(series_random)
    step_d_loss = get_discriminator_loss(X_, Z_)
    if step_d_loss > 0.15:
        step_d_loss = train_discriminator(X_, Z_)

    if step % 1000 == 0:
        print(f'{step:6,.0f} | d_loss: {step_d_loss:6.4f} | g_loss_u: {step_g_loss_u:6.4f} | '
              f'g_loss_s: {g_loss_s_step:6.4f} | g_loss_v: {step_g_loss_v:6.4f} | e_loss_t0: {e_loss_step_start:6.4f}')

    with writer.as_default():
        tf.summary.scalar('G Loss S', g_loss_s_step, step=step)
        tf.summary.scalar('G Loss U', step_g_loss_u, step=step)
        tf.summary.scalar('G Loss V', step_g_loss_v, step=step)
        tf.summary.scalar('E Loss T0', e_loss_step_start, step=step)
        tf.summary.scalar('D Loss', step_d_loss, step=step)

# Save created data
synthetic_data.save(log_dir / 'synthetic_data')

data_created = []
for i in range(int(number_windows / batch_size)):
    Z_ = next(series_random)
    d = synthetic_data(Z_)
    data_created.append(d)

data_created = np.array(np.vstack(data_created))
np.save(log_dir / 'data_created.npy', data_created)
data_created = (scaler.inverse_transform(data_created.reshape(-1, num_sequence)).reshape(-1, sequence_len, num_sequence))
with pd.HDFStore(hdf_store) as store:
    store.put('data/synthetic', pd.DataFrame(data_created.reshape(-1, num_sequence), columns=stock_labels))

# Plot correlations
df = pd.read_hdf(hdf_store, 'data/real')
print(df.head())
df.drop(['AMZN', 'AAPL'], inplace=True, axis=1)

stock_labels2 = ['XOM', 'NDAQ', 'FB', 'GOOGL']

axes = df.div(df.iloc[0]).plot(subplots=False,figsize=(8, 6),legend=stock_labels2)

plt.title('Normalized Prices')
plt.legend(title='')
plt.gcf().tight_layout()

# Plot correlation matrix
plt.style.use('ggplot')
sns.set_style('darkgrid')
sns.set(font_scale=1)
cg = sns.clustermap(df.corr(),annot=True,fmt='.2f',center=-0.4, cmap='inferno');
cg.ax_row_dendrogram.set_visible(False)
cg.ax_col_dendrogram.set_visible(False)


# Plot prediction for a stock
plt.style.use('ggplot')
plt.figure()
index = list(range(1, 25))
synthetic = data_created[np.random.randint(sequence_len)]

idx = 20 # 360:455:10
real = df.iloc[idx: idx + sequence_len]
j = 3
ticker='GOOGL'

plt.plot(range(sequence_len),  real.iloc[:, j].values, 'o-', label="actual")
plt.plot(range(sequence_len), synthetic[:, j], 'x-', label="predicted")
plt.legend()
plt.xlabel('Date')
plt.ylabel('Prices($)')
plt.title('Google')
plt.show()
