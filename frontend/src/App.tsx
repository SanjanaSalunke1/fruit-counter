import { useEffect, useRef, useState } from 'react';

type Box = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  color: 'green' | 'red' | 'yellow' | string;
  color_confidence: number;
  maturity?: 'healthy' | 'unhealthy' | string;
  maturity_confidence?: number;
};

type ColorCounts = Record<string, number>;
type BoxMode = 'maturity' | 'color';

const colorStyles: Record<string, string> = {
  green: '#22c55e',
  red: '#ef4444',
  yellow: '#eab308',
};

const maturityStyles: Record<string, string> = {
  healthy: '#22c55e',
  unhealthy: '#ef4444',
};

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fruitCount, setFruitCount] = useState<number | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [colorCounts, setColorCounts] = useState<ColorCounts | null>(null);
  const [maturityCounts, setMaturityCounts] = useState<ColorCounts | null>(null);
  const [boxMode, setBoxMode] = useState<BoxMode>('maturity');
  const [minConfidence, setMinConfidence] = useState(0.35);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const imgRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setFruitCount(null);
    setBoxes([]);
    setColorCounts(null);
    setMaturityCounts(null);
    setError(null);

    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setImageSize({ width: 0, height: 0 });

    if (file) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    } else {
      setPreviewUrl(null);
    }
  };

  const handleImageLoad = () => {
    if (imgRef.current) {
      setImageSize({
        width: imgRef.current.naturalWidth,
        height: imgRef.current.naturalHeight,
      });
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please choose an image first.');
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`/api/count?min_conf=${minConfidence}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const responseError = await response.json().catch(() => null);
        throw new Error(responseError?.detail ?? `Server error (${response.status}).`);
      }

      const data = await response.json();
      setFruitCount(data.total_bell_pepper_count ?? data.fruit_count);
      setBoxes(data.boxes ?? []);
      setColorCounts(data.color_counts ?? { green: 0, red: 0, yellow: 0 });
      setMaturityCounts(data.maturity_counts ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed. Please try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header>
        <h1>Fruit Counter</h1>
        <p>Upload a bell pepper photo to count peppers by color and maturity.</p>
      </header>

      <main>
        <section className="card">
          <label className="file-input-label">
            <span>Select image</span>
            <input type="file" accept="image/*" onChange={handleFileChange} />
          </label>

          <div className="slider-row">
            <label htmlFor="confidence">Min confidence: {Math.round(minConfidence * 100)}%</label>
            <input
              id="confidence"
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={minConfidence}
              onChange={(event) => setMinConfidence(parseFloat(event.target.value))}
            />
          </div>

          <div className="mode-row" role="group" aria-label="Color boxes by">
            <span>Color boxes by</span>
            {(['maturity', 'color'] as BoxMode[]).map((mode) => (
              <button
                key={mode}
                type="button"
                className={`mode-button${boxMode === mode ? ' active' : ''}`}
                aria-pressed={boxMode === mode}
                onClick={() => setBoxMode(mode)}
              >
                {mode === 'maturity' ? 'Maturity' : 'Color'}
              </button>
            ))}
          </div>

          <button onClick={handleUpload} disabled={!selectedFile || isLoading}>
            {isLoading ? 'Counting...' : 'Count bell peppers'}
          </button>

          {error && <div className="error">{error}</div>}

          {previewUrl && (
            <div className="preview-container">
              <img
                ref={imgRef}
                src={previewUrl}
                alt="Selected bell peppers"
                onLoad={handleImageLoad}
              />
              {imageSize.width > 0 &&
                boxes.map((box, index) => {
                  const left = (box.x1 / imageSize.width) * 100;
                  const top = (box.y1 / imageSize.height) * 100;
                  const width = ((box.x2 - box.x1) / imageSize.width) * 100;
                  const height = ((box.y2 - box.y1) / imageSize.height) * 100;
                  const color = box.color.toLowerCase();
                  const maturity = box.maturity;
                  const colorStyle =
                    boxMode === 'maturity' && maturity
                      ? maturityStyles[maturity] ?? '#38bdf8'
                      : colorStyles[color] ?? '#38bdf8';
                  const label = maturity
                    ? `${maturity} ${Math.round((box.maturity_confidence ?? 0) * 100)}% · ${color}`
                    : color;

                  return (
                    <div
                      key={index}
                      className="box-overlay"
                      style={{
                        left: `${left}%`,
                        top: `${top}%`,
                        width: `${width}%`,
                        height: `${height}%`,
                        borderColor: colorStyle,
                      }}
                    >
                      <span className="box-color-label">
                        <i style={{ backgroundColor: colorStyle }} />
                        {label}
                      </span>
                    </div>
                  );
                })}
            </div>
          )}

          {fruitCount !== null && colorCounts && (
            <div className="result">
              Total bell peppers: <strong>{fruitCount}</strong>
              <div className="count-groups">
                {maturityCounts && (
                  <div>
                    <h3>Maturity</h3>
                    {Object.entries(maturityCounts).map(([maturity, count]) => (
                      <div key={maturity}>
                        <i className="swatch" style={{ backgroundColor: maturityStyles[maturity] ?? '#38bdf8' }} />
                        {maturity}: <strong>{count}</strong>
                      </div>
                    ))}
                  </div>
                )}
                <div>
                  <h3>Color</h3>
                  {Object.entries(colorCounts).map(([color, count]) => (
                    <div key={color} style={{ textTransform: 'capitalize' }}>
                      <i className="swatch" style={{ backgroundColor: colorStyles[color] ?? '#38bdf8' }} />
                      {color}: <strong>{count}</strong>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
