import { useEffect, useRef, useState } from 'react';

type Box = {
  x1: number; y1: number; x2: number; y2: number;
  confidence: number; color: string; color_confidence: number;
};
type ColorCounts = Record<string, number>;
type Activity = { id: number; time: string; message: string; kind: 'info' | 'success' | 'error' };

const colorStyles: Record<string, string> = {
  green: '#438b61', red: '#d96c57', yellow: '#d6a93e',
};

function Icon({ name, size = 20 }: { name: string; size?: number }) {
  const paths: Record<string, string> = {
    leaf: 'M20 4c-8 0-14 4-14 11a5 5 0 0 0 5 5c7 0 11-6 9-16ZM4 20c3-5 6-8 11-11',
    image: 'M4 4h16v16H4zM8.5 10a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3ZM20 15l-5-5L5 20',
    video: 'm15 10 5-3v10l-5-3M4 6h11v12H4z',
    camera: 'M4 8h3l2-3h6l2 3h3v11H4zM12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z',
    scan: 'M4 8V5a1 1 0 0 1 1-1h3m8 0h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3m-8 0H5a1 1 0 0 1-1-1v-3M8 12h8',
    check: 'm5 12 4 4L19 6',
    clock: 'M12 7v5l3 2m6-2a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z',
    arrow: 'M5 12h14m-6-6 6 6-6 6',
    close: 'm18 6-12 12M6 6l12 12',
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name] ?? paths.scan} /></svg>;
}

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
  const [recording, setRecording] = useState(false);
  const [activity, setActivity] = useState<Activity[]>([]);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const photoInputRef = useRef<HTMLInputElement | null>(null);
  const videoInputRef = useRef<HTMLInputElement | null>(null);
  const isVideo = Boolean(selectedFile?.type.startsWith('video/'));

  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);
  useEffect(() => () => { streamRef.current?.getTracks().forEach((track) => track.stop()); }, []);

  const resetResult = () => { setFruitCount(null); setBoxes([]); setColorCounts(null); setError(null); };

  const logActivity = (message: string, kind: Activity['kind'] = 'info') => {
    const now = new Date();
    setActivity((items) => [{
      id: now.getTime() + Math.random(),
      time: now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      message,
      kind,
    }, ...items].slice(0, 8));
  };

  const selectFile = (file: File | null) => {
    resetResult();
    setSelectedFile(file);
    setImageSize({ width: 0, height: 0 });
    setPreviewUrl(file ? URL.createObjectURL(file) : null);
    if (file) logActivity(`${file.type.startsWith('video/') ? 'Video' : 'Photo'} selected: ${file.name}`);
    else logActivity('Selected media removed');
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    setError(null);
    logActivity(`Started ${isVideo ? 'video' : 'photo'} analysis`);
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      const endpoint = isVideo ? '/api/count-video' : '/api/count';
      const response = await fetch(`${endpoint}?min_conf=${minConfidence}`, { method: 'POST', body: formData });
      if (!response.ok) {
        const responseError = await response.json().catch(() => null);
        // Validation errors (422) return a list of {msg}; other errors a string.
        const detail = responseError?.detail;
        const message = Array.isArray(detail) ? detail.map((item) => item.msg).join('; ') : detail;
        throw new Error(message ? `${message}${responseError?.request_id ? ` (request ${responseError.request_id})` : ''}` : `Server error (${response.status}).`);
      }
      const data = await response.json();
      setFruitCount(data.total_bell_pepper_count ?? data.fruit_count);
      setBoxes(data.boxes ?? []);
      setColorCounts(data.color_counts ?? { green: 0, red: 0, yellow: 0 });
      logActivity(`Analysis complete · ${data.total_bell_pepper_count ?? data.fruit_count} fruit detected`, 'success');
    } catch (err) {
      const message = err instanceof TypeError
        ? 'The counting service is offline. Start the backend, then try again.'
        : err instanceof Error ? err.message : 'Analysis failed. Please try again.';
      setError(message);
      logActivity(`Analysis failed · ${message}`, 'error');
    } finally { setIsLoading(false); }
  };

  const toggleRecording = async () => {
    if (recording && recorderRef.current) { recorderRef.current.stop(); setRecording(false); logActivity('Camera recording stopped'); return; }
    resetResult();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      streamRef.current = stream;
      const mimeType = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find((type) => MediaRecorder.isTypeSupported(type));
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorderRef.current = recorder;
      const chunks: BlobPart[] = [];
      recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        const blob = new Blob(chunks, { type: recorder.mimeType || 'video/webm' });
        selectFile(new File([blob], `capture-${Date.now()}.webm`, { type: blob.type }));
      };
      recorder.start(1000);
      setRecording(true);
      logActivity('Camera recording started');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Could not access the camera.';
      setError(message);
      logActivity(`Camera error · ${message}`, 'error');
    }
  };

  const formatBytes = (bytes: number) => bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  const total = fruitCount ?? 0;

  return (
    <div className="app-shell">
      <nav className="topbar">
        <a className="brand" href="#top" aria-label="Fieldnote home"><span className="brand-mark"><Icon name="leaf" size={21} /></span><span>fieldnote<span className="brand-period">.</span></span></a>
        <div className="topbar-right"><span className="workspace-label">PRODUCE INTELLIGENCE</span><span className="service-status"><i /> Counting service</span></div>
      </nav>

      <main id="top" className="workspace">
        <section className="page-heading">
          <div><div className="eyebrow"><span>HARVEST OPERATIONS</span><span className="eyebrow-line" /></div><h1>Fruit counting,<br /><em>without the guesswork.</em></h1><p>Turn a photo or video into a clear, color-by-color inventory.</p></div>
          <div className="heading-note"><span className="note-icon"><Icon name="scan" size={18} /></span><span><strong>AI-assisted inspection</strong><small>Bell pepper detection</small></span></div>
        </section>

        <div className="dashboard-grid">
          <section className="panel source-panel">
            <div className="panel-heading"><div><span className="step-label">01 <i /> INPUT</span><h2>Add your footage</h2><p>Choose a source to begin your count.</p></div><span className="panel-index">SOURCE</span></div>

            <div className="source-options">
              <button className="source-option" type="button" onClick={() => photoInputRef.current?.click()}>
                <span className="option-icon photo-icon"><Icon name="image" /></span><span className="option-copy"><strong>Upload photo</strong><small>JPG, PNG or WEBP</small></span><Icon name="arrow" size={17} />
              </button>
              <input ref={photoInputRef} className="visually-hidden" type="file" accept="image/*" onChange={(event) => selectFile(event.target.files?.[0] ?? null)} />
              <button className="source-option" type="button" onClick={() => videoInputRef.current?.click()}>
                <span className="option-icon video-icon"><Icon name="video" /></span><span className="option-copy"><strong>Upload video</strong><small>MP4, MOV or WEBM</small></span><Icon name="arrow" size={17} />
              </button>
              <input ref={videoInputRef} className="visually-hidden" type="file" accept="video/*" onChange={(event) => selectFile(event.target.files?.[0] ?? null)} />
              <button className={`source-option record-option ${recording ? 'is-recording' : ''}`} type="button" onClick={toggleRecording} disabled={isLoading}>
                <span className="option-icon record-icon"><Icon name={recording ? 'close' : 'camera'} /></span><span className="option-copy"><strong>{recording ? 'Stop recording' : 'Record video'}</strong><small>{recording ? 'Tap to finish capture' : 'Use your device camera'}</small></span>{recording && <span className="record-live"><i /> LIVE</span>}
              </button>
            </div>

            {selectedFile ? (
              <div className="selected-file"><span className="file-badge"><Icon name={isVideo ? 'video' : 'image'} size={18} /></span><span className="file-copy"><strong>{selectedFile.name}</strong><small>{isVideo ? 'Video footage' : 'Photo'} · {formatBytes(selectedFile.size)}</small></span><button className="icon-button" type="button" aria-label="Remove selected file" onClick={() => selectFile(null)}><Icon name="close" size={17} /></button></div>
            ) : (
              <div className="input-hint"><Icon name="check" size={15} /><span>Your media stays in this session and is processed for counting.</span></div>
            )}

            <div className="preview-heading"><span>PREVIEW</span>{selectedFile && <span className="preview-state"><i /> READY TO ANALYZE</span>}</div>
            <div className={`preview-stage ${previewUrl ? 'has-media' : ''}`}>
              {previewUrl ? (
                <>
                  {isVideo ? <video src={previewUrl} controls playsInline /> : <img ref={imgRef} src={previewUrl} alt="Selected fruit for counting" onLoad={() => { if (imgRef.current) setImageSize({ width: imgRef.current.naturalWidth, height: imgRef.current.naturalHeight }); }} />}
                  {!isVideo && imageSize.width > 0 && boxes.map((box, index) => <div key={index} className="box-overlay" style={{ left: `${(box.x1 / imageSize.width) * 100}%`, top: `${(box.y1 / imageSize.height) * 100}%`, width: `${((box.x2 - box.x1) / imageSize.width) * 100}%`, height: `${((box.y2 - box.y1) / imageSize.height) * 100}%`, borderColor: colorStyles[box.color.toLowerCase()] ?? '#438b61' }}><span className="box-color-label">{box.color}</span></div>)}
                </>
              ) : (
                <div className="empty-preview"><span className="empty-art"><span /><span /><span /><i><Icon name="image" size={24} /></i></span><strong>Your preview will appear here</strong><small>Add a photo or video to get started</small></div>
              )}
            </div>

            <div className="settings-row"><div><span className="setting-title">Detection sensitivity</span><small>Higher sensitivity may detect more fruit</small></div><span className="sensitivity-value">{Math.round(minConfidence * 100)}%</span></div>
            <input className="sensitivity-slider" aria-label="Detection sensitivity" type="range" min="0" max="1" step="0.05" value={minConfidence} onChange={(event) => setMinConfidence(parseFloat(event.target.value))} />
            {error && <div className="error-banner"><span>!</span>{error}</div>}
            <button className="analyze-button" onClick={handleUpload} disabled={!selectedFile || isLoading || recording}>
              {isLoading ? <span className="loading-spinner" aria-hidden="true" /> : <Icon name="scan" size={18} />}{isLoading ? 'Analyzing footage…' : isVideo ? 'Analyze video' : 'Analyze photo'}{!isLoading && <Icon name="arrow" size={18} />}
            </button>
            <div className="privacy-note"><span><Icon name="check" size={14} /></span> Secure, session-only analysis</div>
          </section>

          <section className="panel results-panel">
            <div className="panel-heading"><div><span className="step-label">02 <i /> RESULTS</span><h2>Count summary</h2><p>Your inspection at a glance.</p></div><span className={`result-tag ${fruitCount !== null ? 'complete' : ''}`}><i />{fruitCount !== null ? 'COMPLETE' : 'AWAITING INPUT'}</span></div>
            <div className="total-card"><div className="total-card-top"><span>TOTAL FRUIT DETECTED</span><span className="total-spark"><Icon name="scan" size={16} /></span></div><div className="total-number">{fruitCount === null ? '—' : fruitCount}</div><div className="total-caption">{fruitCount === null ? 'Run an analysis to see your count' : isVideo ? 'Unique peppers observed through the camera path' : 'Bell peppers detected in this photo'}</div><div className="total-divider" /><div className="total-foot"><span><i /> {fruitCount !== null ? 'Analysis complete' : 'Ready when you are'}</span>{fruitCount !== null && <span>THIS SESSION</span>}</div></div>

            <div className="breakdown-heading"><div><h3>Color breakdown</h3><p>Sorted by detected color</p></div><span className="breakdown-count">{fruitCount === null ? '—' : `${total} total`}</span></div>
            <div className="color-list">{['green', 'red', 'yellow'].map((color) => {
              const count = colorCounts?.[color] ?? 0;
              const pct = total ? (count / total) * 100 : 0;
              return <div className="color-row" key={color}><div className="color-row-top"><span className="color-name"><i style={{ backgroundColor: colorStyles[color] }} />{color}</span><span className="color-value">{fruitCount === null ? '—' : count}<small>{fruitCount === null ? '' : ` · ${Math.round(pct)}%`}</small></span></div><div className="color-track"><span style={{ width: `${pct}%`, backgroundColor: colorStyles[color] }} /></div></div>;
            })}</div>

            <div className="method-card"><span className="method-icon"><Icon name="scan" size={18} /></span><div><strong>How we count</strong><p>{isVideo ? 'Camera motion is compensated so repeat sightings of peppers along the scan count once.' : 'Detected fruit is counted once in the selected image.'}</p></div></div>
            <div className="activity-section">
              <div className="activity-heading"><h3>Activity log</h3><span>{isLoading ? <><i className="activity-pulse" /> RUNNING</> : 'THIS SESSION'}</span></div>
              {activity.length ? <div className="activity-list">{activity.map((item) => <div className={`activity-item ${item.kind}`} key={item.id}><span className="activity-dot" /><time>{item.time}</time><p>{item.message}</p></div>)}</div> : <p className="activity-empty">Analysis events and errors will appear here.</p>}
            </div>
            <div className="results-footer"><span><Icon name="clock" size={14} /> RESULTS UPDATE AFTER ANALYSIS</span><span>FIELDNOTE <b>·</b> 01</span></div>
          </section>
        </div>
        <footer className="page-footer"><span>FIELDNOTE <i>·</i> PRODUCE INTELLIGENCE</span><span>BUILT FOR THE FIELD</span></footer>
      </main>
    </div>
  );
}

export default App;
