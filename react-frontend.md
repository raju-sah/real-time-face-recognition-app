# skill: react-frontend

## description

Builds a React + Tailwind frontend for interacting with the face recognition FastAPI backend.

---

## workflow

### 1. SETUP

* Use Vite
* Package manager: pnpm

Commands:
pnpm create vite frontend
pnpm install

---

### 2. INSTALL DEPENDENCIES

* axios
* tailwindcss

---

### 3. PROJECT STRUCTURE

src/
components/
pages/
services/
hooks/

---

### 4. UI FEATURES

#### Upload Component

* Drag & drop or file input
* Preview image

#### Recognize Button

* Send image to backend

#### Result Display

* Show:

  * Person name
  * Confidence score
  * Image preview

---

### 5. API INTEGRATION

* Use axios
* Base URL: http://localhost:8000

---

### 6. STATE HANDLING

* loading state
* error state
* result state

---

### 7. STYLING

* TailwindCSS
* Clean, centered layout
* Responsive design

---

## constraints

* Do NOT hardcode API URLs
* Do NOT block UI during request
* Handle API errors properly

---

## validation

* Test file upload flow
* Verify API response handling
* Ensure UI updates correctly
