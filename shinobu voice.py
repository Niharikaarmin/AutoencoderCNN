import tensorflow as tf
import numpy as np
import librosa
import os
import soundfile as sf
from tensorflow.keras.layers import Input, LSTM, Dense, Conv1D, BatchNormalization
from tensorflow.keras.models import Model

# Constants
SAMPLE_RATE = 22050
N_MELS = 80
FFT_SIZE = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
MAX_TEXT_LEN = 100
EMBEDDING_DIM = 512
BATCH_SIZE = 32

# Preprocessing: Convert text to sequence of integers
def text_to_sequence(text, char2idx):
    return [char2idx.get(c, 0) for c in text.lower()]

# Preprocessing: Convert audio to mel spectrogram
def audio_to_mel(audio_file):
    try:
        audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=N_MELS, n_fft=FFT_SIZE, hop_length=HOP_LENGTH, win_length=WIN_LENGTH
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        return mel.T  # Shape: (time_steps, n_mels)
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

# Load transcriptions from a file
def load_transcriptions(transcription_file):
    transcriptions = {}
    if not os.path.exists(transcription_file):
        raise FileNotFoundError(
            f"Transcription file {transcription_file} not found. Create a 'transcriptions.txt' file with format: filename|transcription")
    with open(transcription_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                filename, text = line.strip().split('|')
                transcriptions[filename] = text
            except ValueError:
                print(f"Invalid line in {transcription_file}: {line.strip()}")
    return transcriptions

# Data generator for training
def data_generator(data_dir, char2idx):
    audio_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]
    transcription_file = os.path.join(data_dir, 'transcriptions.txt')
    print(f"Found {len(audio_files)} .wav files in {data_dir}")

    transcriptions = load_transcriptions(transcription_file)
    print(f"Loaded {len(transcriptions)} transcriptions from {transcription_file}")

    # Verify that all audio files have transcriptions
    audio_filenames = [os.path.basename(f) for f in audio_files]
    missing_transcriptions = [f for f in audio_filenames if f not in transcriptions]
    if missing_transcriptions:
        raise ValueError(f"Missing transcriptions for files: {missing_transcriptions}")

    if not audio_files:
        raise ValueError(f"No .wav files found in {data_dir}")

    while True:
        idx = np.random.randint(0, len(audio_files))
        audio_file = audio_files[idx]
        filename = os.path.basename(audio_file)
        text = transcriptions[filename]
        mel = audio_to_mel(audio_file)
        if mel is None:
            continue  # Skip invalid audio files
        text_seq = text_to_sequence(text, char2idx)
        text_seq = tf.keras.preprocessing.sequence.pad_sequences(
            [text_seq], maxlen=MAX_TEXT_LEN, padding='post'
        )[0]
        yield (text_seq, mel), mel

# Build Tacotron 2-like model
def build_tacotron_model(vocab_size):
    # Encoder
    text_input = Input(shape=(MAX_TEXT_LEN,), name='text_input')
    x = tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM)(text_input)
    x = Conv1D(filters=512, kernel_size=5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x, state_h, state_c = LSTM(512, return_sequences=True, return_state=True)(x)

    # Attention mechanism
    attention = tf.keras.layers.Attention()([x, x])

    # Decoder
    mel_input = Input(shape=(None, N_MELS), name='mel_input')
    y = LSTM(512, return_sequences=True)(mel_input, initial_state=[state_h, state_c])
    y = tf.keras.layers.Concatenate()([y, attention])
    y = LSTM(512, return_sequences=True)(y)
    mel_output = Dense(N_MELS)(y)

    model = Model(inputs=[text_input, mel_input], outputs=mel_output)
    return model

# Vocoder: Simple Griffin-Lim for mel to audio conversion
def mel_to_audio(mel):
    mel = librosa.db_to_power(mel)
    audio = librosa.feature.inverse.mel_to_audio(
        mel, sr=SAMPLE_RATE, n_fft=FFT_SIZE, hop_length=HOP_LENGTH, win_length=WIN_LENGTH
    )
    return audio

# Training setup
def train_model(data_dir, epochs=10):
    # Character mapping
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789,.!? '
    char2idx = {c: i + 1 for i, c in enumerate(chars)}
    vocab_size = len(chars) + 1
    print(f"Vocabulary size: {vocab_size}")

    # Build and compile model
    model = build_tacotron_model(vocab_size)
    model.compile(optimizer='adam', loss='mse')
    print("Model compiled successfully")

    # Data generator
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(data_dir, char2idx),
        output_signature=(
            (tf.TensorSpec(shape=(MAX_TEXT_LEN,), dtype=tf.int32),
             tf.TensorSpec(shape=(None, N_MELS), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, N_MELS), dtype=tf.float32)
        )
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Train
    model.fit(dataset, epochs=epochs, steps_per_epoch=100)  # Adjust steps_per_epoch based on dataset size

    return model, char2idx

# Inference: Generate speech from text
def generate_speech(text, model, char2idx):
    text_seq = text_to_sequence(text, char2idx)
    text_seq = tf.keras.preprocessing.sequence.pad_sequences(
        [text_seq], maxlen=MAX_TEXT_LEN, padding='post'
    )
    # Dummy mel input for initial decoding
    dummy_mel = np.zeros((1, 1, N_MELS), dtype=np.float32)
    mel_pred = model.predict([text_seq, dummy_mel])
    audio = mel_to_audio(mel_pred[0].T)
    return audio

# Main execution
if __name__ == '__main__':
    # Path to dataset
    data_dir = r'C:\Users\jenit\PycharmProjects\lstmtrain\Ai shinobu voice'

    # Train model
    try:
        model, char2idx = train_model(data_dir, epochs=10)

        # Generate sample speech
        sample_text = "Hello, I am Shinobu Kocho."
        audio = generate_speech(sample_text, model, char2idx)

        # Save output
        sf.write('outputshin.wav', audio, SAMPLE_RATE)
        print("Speech generated and saved as 'outputshin.wav'")
    except Exception as e:
        print(f"Error during execution: {e}")