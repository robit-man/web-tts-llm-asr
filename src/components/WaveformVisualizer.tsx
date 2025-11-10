import { useEffect, useRef } from 'react';

interface WaveformVisualizerProps {
  audioBuffer: AudioBuffer | null;
  width?: number;
  height?: number;
}

export default function WaveformVisualizer({
  audioBuffer,
  width = 800,
  height = 200
}: WaveformVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !audioBuffer) {
      // Clear canvas if no audio
      if (canvas) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.fillStyle = 'transparent';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.fillStyle = '#0ea5e9';
          ctx.font = '14px monospace';
          ctx.textAlign = 'center';
          ctx.fillText('No audio data', canvas.width / 2, canvas.height / 2);
        }
      }
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = 'transparent';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Get audio data from first channel
    const data = audioBuffer.getChannelData(0);
    const step = Math.ceil(data.length / width);
    const amp = height / 2;

    // Draw waveform
    ctx.strokeStyle = '#0ea5e9';
    ctx.lineWidth = 1;
    ctx.beginPath();

    for (let i = 0; i < width; i++) {
      const min = Math.min(...Array.from({ length: step }, (_, j) => data[i * step + j] || 0));
      const max = Math.max(...Array.from({ length: step }, (_, j) => data[i * step + j] || 0));

      ctx.moveTo(i, (1 + min) * amp);
      ctx.lineTo(i, (1 + max) * amp);
    }

    ctx.stroke();

    // Draw center line
    ctx.strokeStyle = '#444';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Draw info text
    ctx.fillStyle = '#0ea5e9';
    ctx.font = '12px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(
      `Duration: ${audioBuffer.duration.toFixed(2)}s | ` +
      `Sample Rate: ${audioBuffer.sampleRate}Hz | ` +
      `Samples: ${data.length}`,
      10,
      20
    );

    // Calculate and display RMS
    let sumSquares = 0;
    for (let i = 0; i < data.length; i++) {
      sumSquares += data[i] * data[i];
    }
    const rms = Math.sqrt(sumSquares / data.length);
    ctx.fillText(`RMS Level: ${(rms * 100).toFixed(2)}%`, 10, 40);

  }, [audioBuffer, width, height]);

  return (
    <div style={{
      background: '#020617a6',
      borderRadius: '8px',
      border: '1px solid #94a3b84d'
    }}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          display: 'block',
          width: '100%',
          height: 'auto',
          maxWidth: `${width}px`
        }}
      />
    </div>
  );
}
