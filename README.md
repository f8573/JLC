# LambdaCompute (JLAC)

LambdaCompute is a Java linear algebra library and web application for matrix diagnostics, decomposition, and spectral analysis.

Live site: https://lambdacompute.org/

## What this project includes

- `src/main/java/net/faulj`: Java core library (matrix/vector types, decompositions, solvers, eigen/spectral routines, condition/accuracy metrics, benchmarking helpers)
- `src/main/java/net/faulj/web`: Spring Boot API layer (`/api/diagnostics`, `/api/status`, `/api/contact`, benchmark/status streams)
- `frontend`: React + Vite client for matrix input, analysis views, decompositions, spectral reports, favorites/history, and settings

## Primary use cases

- Run matrix diagnostics from raw matrix input
- Inspect decomposition results (QR, LU, SVD, Schur, Hessenberg, spectral, etc.)
- Evaluate numerical stability and accuracy metadata
- Benchmark selected compute paths via API endpoints

## Requirements

- Java 21 (project toolchain target)
- Node.js 18+
- npm 9+

## Run locally

### 1) Start backend (Spring Boot)

From repository root:

```powershell
.\gradlew.bat bootRun
```

Backend default: `http://localhost:8080`

### 2) Start frontend (Vite)

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend default: `http://localhost:5173`

The Vite dev server proxies `/api` to `http://localhost:8080` via `frontend/vite.config.js`.

## Build

### Backend

```powershell
.\gradlew.bat build
```

### Frontend

```powershell
cd frontend
npm run build
```

## Test

```powershell
.\gradlew.bat test
```

If you only want a targeted API smoke test:

```powershell
.\gradlew.bat test --tests net.faulj.web.ApiControllerStatusTest
```

## API at a glance

- `GET /api/ping`
- `GET /api/status`
- `POST /api/diagnostics`
- `GET /api/diagnostics?matrix=...`
- `GET /api/diagnostics/stream` (SSE)
- `GET /api/benchmark/diagnostic`
- `GET /api/benchmark/diagnostic512`
- `POST /api/contact`

## Notes

- Contact form delivery uses `DISCORD_WEBHOOK_URL` from environment variables.
- Large matrices are intentionally limited for synchronous full diagnostics in the API.

## License

See `LICENSE`.
