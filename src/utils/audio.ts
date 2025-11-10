export const TARGET_SAMPLE_RATE = 16000;

export function audioBufferToFloat32(buffer: AudioBuffer) {
  const mono = convertToMono(buffer);
  if (buffer.sampleRate === TARGET_SAMPLE_RATE) {
    return mono;
  }
  return resampleToRate(mono, buffer.sampleRate, TARGET_SAMPLE_RATE);
}

export function convertToMono(buffer: AudioBuffer) {
  if (buffer.numberOfChannels === 1) {
    return buffer.getChannelData(0).slice(0);
  }

  const left = buffer.getChannelData(0);
  const right = buffer.getChannelData(1);
  const mono = new Float32Array(buffer.length);

  for (let i = 0; i < buffer.length; i++) {
    mono[i] = (left[i] + right[i]) * 0.5;
  }

  return mono;
}

export function resampleToRate(
  data: Float32Array,
  sourceSampleRate: number,
  targetSampleRate: number,
) {
  if (sourceSampleRate === targetSampleRate) {
    return data;
  }

  const sampleRatio = sourceSampleRate / targetSampleRate;
  const newLength = Math.round(data.length / sampleRatio);
  const result = new Float32Array(newLength);

  for (let i = 0; i < newLength; i++) {
    const position = i * sampleRatio;
    const leftIndex = Math.floor(position);
    const rightIndex = Math.min(data.length - 1, leftIndex + 1);
    const interpolate = position - leftIndex;
    const sample =
      (1 - interpolate) * data[leftIndex] + interpolate * data[rightIndex];
    result[i] = sample;
  }

  return result;
}
