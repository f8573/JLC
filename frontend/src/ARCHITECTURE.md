# React Component Architecture

## Overview
This project follows **Atomic Design principles** for a fully modular React component architecture.

## Directory Structure

```
src/
├── components/
│   ├── ui/              # Atomic components (primitives)
│   │   ├── Logo.jsx
│   │   ├── IconButton.jsx
│   │   ├── SearchBar.jsx
│   │   ├── UserAvatar.jsx
│   │   ├── Badge.jsx
│   │   ├── Button.jsx
│   │   ├── NumberInput.jsx
│   │   └── index.js
│   │
│   ├── matrix/          # Molecular components (matrix-specific)
│   │   ├── MatrixCell.jsx
│   │   ├── MatrixGrid.jsx
│   │   ├── MatrixDimensionControl.jsx
│   │   ├── MatrixActions.jsx
│   │   ├── MatrixInputCard.jsx
│   │   └── index.js
│   │
│   ├── features/        # Feature components
│   │   ├── FeatureCard.jsx
│   │   ├── FeatureGrid.jsx
│   │   └── index.js
│   │
│   ├── results/         # Results-specific components
│   │   ├── PropertyCard.jsx
│   │   ├── SummaryItem.jsx
│   │   ├── Breadcrumb.jsx
│   │   └── index.js
│   │
│   ├── Header.jsx       # Organism components
│   ├── Sidebar.jsx
│   ├── MatrixHeader.jsx
│   ├── MatrixSidebar.jsx
│   ├── MatrixInput.jsx
│   └── MatrixResults.jsx
│
├── hooks/               # Custom React hooks
│   ├── useMatrix.js
│   ├── useMatrixAnimation.js
│   └── index.js
│
├── pages/               # Page templates
│   ├── MainPage.jsx
│   └── MatrixPage.jsx
│
└── App.jsx              # Root component & router

```

## Design Principles

### 1. **Atomic Components** (`ui/`)
Smallest, reusable building blocks that can be used anywhere:
- `Logo` - Brand logo with variants
- `IconButton` - Icon-only button with multiple variants
- `SearchBar` - Input with search icon
- `UserAvatar` - User profile image
- `Badge` - Status/label badges with animation support
- `Button` - Primary action buttons with icon support
- `NumberInput` - Numeric input with constraints

### 2. **Molecular Components** (`matrix/`, `features/`, `results/`)
Combinations of atomic components for specific purposes:
- **Matrix**: Cell inputs, grids, dimension controls, action buttons
- **Features**: Feature cards and grids
- **Results**: Property cards, summary items, breadcrumbs

### 3. **Organism Components** (root level)
Complete sections combining molecular/atomic components:
- `Header`, `Sidebar` - Layout components
- `MatrixInput`, `MatrixResults` - Major feature sections

### 4. **Page Templates** (`pages/`)
Full pages combining organisms:
- `MainPage` - Landing/input page
- `MatrixPage` - Results/analysis page

### 5. **Custom Hooks** (`hooks/`)
Reusable stateful logic:
- `useMatrix` - Matrix state management (CRUD operations)
- `useMatrixAnimation` - Animation logic for transpose

## Usage Examples

### Import from index files:
```jsx
import { Logo, Button, Badge } from './components/ui'
import { MatrixGrid, MatrixActions } from './components/matrix'
import { useMatrix, useMatrixAnimation } from './hooks'
```

### Using atomic components:
```jsx
<Button variant="primary" icon="analytics" onClick={handleClick}>
  Analyze Matrix
</Button>
```

### Using custom hooks:
```jsx
const { rows, cols, values, updateCell, transpose } = useMatrix(2, 2)
const { containerRef, animateTranspose } = useMatrixAnimation()
```

## Benefits

1. **Reusability**: Atomic components can be used throughout the app
2. **Maintainability**: Changes to primitives cascade automatically
3. **Testability**: Small components are easy to unit test
4. **Scalability**: Add features by composing existing components
5. **Consistency**: Design system enforced through shared components
6. **Performance**: Optimized re-renders with proper component boundaries
7. **Developer Experience**: Clear structure, easy to navigate

## Adding New Components

1. **Identify level**: Is it atomic, molecular, or organism?
2. **Create file**: Place in appropriate directory
3. **Export**: Add to corresponding `index.js`
4. **Document**: Add props/usage to this README if needed

## Component Props Pattern

All components follow consistent prop patterns:
- `variant` for style variations
- `className` for custom styling
- `onClick` for actions
- `children` for composition
- Clear, semantic prop names
