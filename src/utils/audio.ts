export const TARGET_SAMPLE_RATE = 16000;

export function audioBufferToFloat32(buffer: AudioBuffer) {
  console.log('[Audio Utils] audioBufferToFloat32 input:', {
    duration: buffer.duration,
    sampleRate: buffer.sampleRate,
    length: buffer.length,
    numberOfChannels: buffer.numberOfChannels
  });

  const mono = convertToMono(buffer);
  console.log('[Audio Utils] After convertToMono:', {
    length: mono.length,
    rms: Math.sqrt(mono.reduce((sum, val) => sum + val * val, 0) / mono.length)
  });

  if (buffer.sampleRate === TARGET_SAMPLE_RATE) {
    console.log('[Audio Utils] Sample rate matches, no resampling needed');
    return mono;
  }

  const resampled = resampleToRate(mono, buffer.sampleRate, TARGET_SAMPLE_RATE);
  console.log('[Audio Utils] After resampleToRate:', {
    originalSampleRate: buffer.sampleRate,
    targetSampleRate: TARGET_SAMPLE_RATE,
    originalLength: mono.length,
    resampledLength: resampled.length,
    rms: Math.sqrt(resampled.reduce((sum, val) => sum + val * val, 0) / resampled.length)
  });

  return resampled;
}

export function convertToMono(buffer: AudioBuffer) {
  if (buffer.numberOfChannels === 1) {
    return buffer.getChannelData(0).slice(0);
  }

  const left = buffer.getChannelData(0);
  const right = buffer.getChannelData(1);
  const length = Math.min(left.length, right.length);
  const mono = new Float32Array(length);
  const scalingFactor = Math.sqrt(2);

  for (let i = 0; i < length; i++) {
    mono[i] = (scalingFactor * (left[i] + right[i])) / 2;
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
