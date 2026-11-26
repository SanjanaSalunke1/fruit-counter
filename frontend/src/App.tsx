import { useEffect, useRef, useState } from 'react';

type Box = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
  color: 'green' | 'red' | 'yellow' | string;
  color_confidence: number;
};

type ColorCounts = Record<string, number>;

const colorStyles: Record<string, string> = {
  green: '#22c55e',
  red: '#ef4444',
  yellow: '#eab308',
};

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fruitCount, setFruitCount] = useState<number | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [colorCounts, setColorCounts] = useState<ColorCounts | null>(null);
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
        <p>Upload a bell pepper photo to count peppers by color.</p>
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
                  const colorStyle = colorStyles[color] ?? '#38bdf8';

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
                        {color}
                      </span>
                    </div>
                  );
                })}
            </div>
          )}

          {fruitCount !== null && colorCounts && (
            <div className="result">
              Total bell peppers: <strong>{fruitCount}</strong>
              {Object.entries(colorCounts).map(([color, count]) => (
                <div key={color} style={{ textTransform: 'capitalize' }}>
                  {color}: <strong>{count}</strong>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;
