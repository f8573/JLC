# JLAC — Java Linear Algebra Calculator

JLAC is a Java-first linear algebra library with a modern web interface for matrix exploration, diagnostics, and decomposition workflows. It provides a rich set of computational routines (LU/QR/SVD/eigen, condition numbers, norms, inverses, and more) while pairing those algorithms with a responsive frontend for interactive matrix analysis.

This repository contains:
- A Java computation core in [src/main/java/net/faulj](src/main/java/net/faulj)
- A Vite/React frontend in [frontend](frontend)
- A thin web layer for API exposure in [src/main/java/net/faulj/web](src/main/java/net/faulj/web)

## Progress report

**Core:** broad algorithm coverage for decomposition, solver, and matrix utilities; additional test coverage and benchmarking are in progress.

**Frontend:** matrix explorer UI with tabs for structure, decomposition, spectral views, and reporting; ongoing polish and UX refinement.

**Web API:** Spring-based application skeleton and CORS configuration present; endpoints continue to expand with additional diagnostics.

## Usage

### Java library (core)
Use JLAC as a standard Java dependency and call algorithms directly on `Matrix` and `Vector` objects. The API follows conventional linear algebra naming (e.g., `LUDecomposition`, `SVDecomposition`, `MatrixNorms`) and returns strongly typed results for factorization outputs.

### Web UI
The frontend is a Vite/React app that consumes the web API. It offers matrix input, diagnostics, and visualization for decomposition, spectral properties, and reporting.

## Installation

### Prerequisites
- Java 17+ (recommended)
- Node.js 18+ (for frontend)

### Backend (Gradle)
1. Build the core library:
  - `./gradlew build`
2. Run tests (if configured):
  - `./gradlew test`

### Frontend (Vite)
1. Install dependencies in [frontend](frontend):
  - `npm install`
2. Start the dev server:
  - `npm run dev`

## Examples

### Compute a QR decomposition
Create a `Matrix`, call a decomposition, and inspect the result objects for $Q$ and $R$ factors.

### Estimate a condition number
Use `ConditionNumber` and `MatrixNorms` to obtain condition estimates for diagnostics.

### Run SVD
Use `SVDecomposition` or `RandomizedSVD` for full or approximate SVD depending on size.

## License

**MIT License with Attribution Requirement.**

You may use, modify, and distribute this project, including in commercial work, provided that attribution to the original author (James Faul) is preserved in documentation, credits, or an about page.

See [LICENSE](LICENSE) for the full text.