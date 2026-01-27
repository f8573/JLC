# Frontend Architecture

## Overview
This frontend is organized around **Atomic Design** with clear separation between
UI primitives, feature-level matrix components, layout shells, and route-level pages.
State and computation are encapsulated in custom hooks, while utilities centralize
formatting, navigation, and diagnostics helpers.

## Directory Structure

```
src/
├── App.jsx                      # Root component
├── main.jsx                     # App bootstrap
├── ARCHITECTURE.md              # This document
├── components/
│   ├── ui/                       # Atomic primitives
│   │   ├── Badge.jsx
│   │   ├── Button.jsx
│   │   ├── IconButton.jsx
│   │   ├── Latex.jsx
│   │   ├── Logo.jsx
│   │   ├── NumberInput.jsx
│   │   ├── SearchBar.jsx
│   │   ├── UserAvatar.jsx
│   │   └── index.js
│   ├── matrix/                   # Matrix-specific molecular components
│   │   ├── MatrixActions.jsx
│   │   ├── MatrixCell.jsx
│   │   ├── MatrixDimensionControl.jsx
│   │   ├── MatrixDisplay.jsx
│   │   ├── MatrixGrid.jsx
│   │   ├── MatrixInputCard.jsx
│   │   └── index.js
│   ├── features/                 # Feature cards and grids
│   │   ├── FeatureCard.jsx
│   │   ├── FeatureGrid.jsx
│   │   └── index.js
│   ├── results/                  # Result summary blocks
│   │   ├── Breadcrumb.jsx
│   │   ├── PropertyCard.jsx
│   │   ├── SummaryItem.jsx
│   │   └── index.js
│   ├── layout/                   # Page-level layout helpers
│   │   ├── MatrixAnalysisLayout.jsx
│   │   └── MatrixTabs.jsx
│   ├── Header.jsx                # Organisms
│   ├── MatrixHeader.jsx
│   ├── MatrixInput.jsx
│   ├── MatrixResults.jsx
│   ├── MatrixSidebar.jsx
│   └── Sidebar.jsx
├── hooks/                        # State & computation logic
│   ├── useDiagnostics.js
│   ├── useMatrix.js
│   ├── useMatrixAnimation.js
│   ├── useMatrixCompute.js
│   └── index.js
├── pages/                        # Route-level pages
│   ├── FavoritesPage.jsx
│   ├── HistoryPage.jsx
│   ├── MainPage.jsx
│   ├── MatrixBasicPage.jsx
│   ├── MatrixDecomposePage.jsx
│   ├── MatrixPage.jsx
│   ├── MatrixReportPage.jsx
│   ├── MatrixSpectralPage.jsx
│   ├── MatrixStructurePage.jsx
│   ├── RecentPage.jsx
│   └── SettingsPage.jsx
└── utils/                        # Pure helpers
    ├── diagnostics.js
    ├── format.js
    ├── navigation.js
    └── spectralSeverity.js
```

## Design Principles

### 1) UI Primitives (components/ui)
Reusable, style-focused atoms used across the app.
- Buttons, icons, badges, and numeric inputs
- `Latex` for KaTeX rendering of equations
- `Logo` and `UserAvatar` for identity elements

### 2) Molecular Components (components/matrix, components/features, components/results)
Feature-oriented blocks composed from UI primitives.
- Matrix grid and cell input controls
- Matrix actions and dimension controls
- Feature highlight cards and result summary items

### 3) Organisms + Layout (components/*, components/layout)
High-level page sections and layout shells that assemble molecular components.
- `Header`, `Sidebar`, `MatrixSidebar`
- `MatrixInput`, `MatrixResults`
- `MatrixAnalysisLayout` and `MatrixTabs` for consistent page structure

### 4) Pages (pages)
Route-level views that compose organisms and connect to hooks.
- Matrix flows: basic, decomposition, spectral, structure, report
- Supporting views: favorites, history, recent, settings

### 5) Hooks (hooks)
Reusable stateful logic and computation interfaces.
- `useMatrix`: matrix values, dimensions, and editing actions
- `useMatrixCompute`: decomposition/eigen/etc. compute orchestration
- `useDiagnostics`: validation and diagnostics collection
- `useMatrixAnimation`: UI animation state for transforms

### 6) Utilities (utils)
Pure functions for formatting, navigation helpers, and diagnostic metadata.

## Data & UI Flow
1. Pages wire hooks to feature components.
2. Matrix inputs update state via `useMatrix` and trigger compute via `useMatrixCompute`.
3. Diagnostics flow through `useDiagnostics` into results views.
4. Formatting and navigation helpers keep page logic lean.

## Usage Examples

### Imports from index files
```jsx
import { Button, Badge, Latex } from './components/ui'
import { MatrixGrid, MatrixActions } from './components/matrix'
import { useMatrix, useMatrixCompute } from './hooks'
```

### Component usage
```jsx
<Button variant="primary" onClick={handleCompute}>
  Analyze Matrix
</Button>
```

### Hook usage
```jsx
const { rows, cols, values, updateCell, transpose } = useMatrix(2, 2)
const { compute, isRunning, results } = useMatrixCompute()
```

## Adding New Components
1. Choose the layer: UI, matrix/feature/results, layout, or page.
2. Create the file and export via the local `index.js` if applicable.
3. Add JSDoc for props, events, and return values.
4. Keep components pure and delegate logic to hooks/utilities.

## Component Props Conventions
- `variant`: visual style option
- `className`: styling override hooks
- `onClick`, `onChange`: event handlers
- `children`: composition slot
- Use semantic, self-describing prop names
