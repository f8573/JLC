# JLC

Welcome to the Java Linear Calculator! This Java project provides exact numerical manipulation for real and complex
numbers, ensuring no loss of precision (except when approximating to decimal formats). The library leverages these exact
numerical operations to offer robust linear algebra functionalities.

## Features

- [x] Ring and field structure to allow types of number systems (e.g. matrices with noncommutative multiplication)
- [ ] Numbers
    - [x] Integers
    - [x] Rational Numbers
    - [ ] Irrational Numbers
        - [ ] Square root
            - [ ] Addition
            - [x] Multiplication
    - [ ] Complex Numbers
- [ ] Linear Algebra
    - [ ] Vectors
    - [ ] Matrices

    ## Frontend + Backend (development)

    This repository now includes a minimal Spring Boot backend and a Vite + React frontend for development.

    - Backend: run the Spring Boot app (port 8080)
    - Frontend: run the Vite dev server (port 5173) which proxies `/api` to the backend.

    Quick start:

    1. Start the Java backend:

    ```powershell
    ./gradlew.bat bootRun
    ```

    2. In a separate terminal, start the frontend dev server:

    ```bash
    cd frontend
    npm install
    npm run dev
    ```

    3. Open http://localhost:5173 to view the frontend which will call the backend `/api/ping` endpoint.

