import torch
import torch.nn as nn
import numpy as np
import torchaudio


def numpy_to_tensor(waveform_np):
    """
    Converts a NumPy waveform array to a PyTorch tensor.
    """
    return torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)


def ctc_loss_numpy(waveform_np, target_text, n_mels=40, num_classes=29):
    """
    Calculates the CTC loss for a given waveform (NumPy array) and target text.

    Args:
        waveform_np (np.ndarray): The waveform as a NumPy array.
        target_text (str): The desired transcript.
        n_mels (int): Number of mel bands. Default is 40.
        num_classes (int): Number of output classes (including blank). Default is 29.

    Returns:
        float: The computed CTC loss.
    """
    # Convert the NumPy waveform to a PyTorch tensor
    waveform = numpy_to_tensor(waveform_np)

    # Extract Mel-spectrogram features
    mel_transform = torchaudio.transforms.MelSpectrogram(n_mels=n_mels)
    mel_spectrogram = mel_transform(waveform)

    # Encode the target text into numeric format
    def text_to_numbers(text):
        alphabet_map = {chr(i + 96): i for i in range(1, 27)}  # a=1, b=2, ..., z=26
        alphabet_map[' '] = 27  # Map space to 27
        return [alphabet_map[char] for char in text.lower() if char in alphabet_map]

    transcript = text_to_numbers(target_text)
    target_sequence = torch.tensor(transcript, dtype=torch.long)

    # Create random logits as placeholders (replace with real model output in practice)
    batch_size = 1
    time_steps = mel_spectrogram.shape[-1]
    logits = torch.randn(time_steps, batch_size, num_classes).log_softmax(2)

    # Define input lengths and target lengths
    input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long)
    target_lengths = torch.tensor([len(target_sequence)], dtype=torch.long)

    # Define CTC Loss
    ctc_loss_fn = nn.CTCLoss()

    # Compute the CTC loss
    loss = ctc_loss_fn(logits, target_sequence.unsqueeze(0), input_lengths, target_lengths)
    return loss.item()


# Example usage
if __name__ == "__main__":
    # Simulated example: Replace with actual NumPy waveform and transcript
    example_waveform = np.random.randn(16000)  # 1 second of random audio data at 16kHz
    example_transcript = "hello world"
    loss = ctc_loss_numpy(example_waveform, example_transcript)
    print(f"CTC Loss: {loss}")
