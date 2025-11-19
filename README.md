# 📦 Installation

## Clone the project

```bash
git clone https://github.com/AliSharjeell/TheVisionariesFinal
```

# Backend Installation + Run

## 📥 Install Dependencies

---

### 1. Go to folder:

```bash
cd TheVisionariesFinal
cd TheVisionariesBackend
```

### 2. Install requirements:

Install **requirements**:

```bash
pip install -r requirements.txt

```

### 3. (Important) Install PyTorch manually if needed

If SentenceTransformer errors, install PyTorch CPU-only:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu

```

---

# 🔑 Environment Variables

add  ur gemini key to `.env` folder:  

```
GEMINI_API_KEY=YOUR_API_KEY_HERE
```

Get your key from:

https://aistudio.google.com/

---

# ▶️ Running the Server

### Start at **0.0.0.0** so mobile apps can connect:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

```

---

# Frontend Installation + Run (Only After Backend)

# 🔁 1. Pre-steps (from your terminal)

```bash

cd TheVisionariesFrontend

```

---

# 🧰 2. Requirements

- Node.js (recommended v18 - v20). Check with:
    
    ```bash
    node -v
    ```
    
- npm (comes with Node) or Yarn (optional).
    
    ```bash
    npm -v
    # or
    yarn -v
    ```
    
- Expo CLI (local recommended):
    
    ```bash
    npm install -g expo-cli
    # or use the new local CLI:
    npm install -g expo
    
    ```
    

---

# 📦 3. Install project dependencies

From `TheVisionariesFrontend`:

```bash
# using npm
npm install

# or using yarn
yarn install

```

---

# 🖥️ 4. Find your machine IPv4 address (Windows)

1. Open Command Prompt and run:
    
    ```bash
    ipconfig
    ```
    
2. Find the network adapter you use (Wi-Fi or Ethernet) and look for `IPv4 Address`. It looks like `192.168.x.y` or `10.0.x.y`. That is the address you will use.

> If you're on macOS / Linux use:
> 

```bash
# macOS
ipconfig getifaddr en0  # common for Wi-Fi
# or
ifconfig

```

---

# ✏️ 5. Replace the IP attribute in the app

The app contains an **IP attribute** (used for the backend URL / sockets / API). You must replace it with your machine IPv4 found above.

1. Open the project in your editor (VSCode/sublime/etc).
2. Press `Ctrl+F` (or `Cmd+F` on macOS) and search for common keys:
    - `IP`
    - or the actual IP currently in the files (e.g., `192.168.0.2`)
3. Replace the value with your IPv4 address. Example replacements:

If you see:

```jsx
const IP = "192.168.0.3";
```

change to:

```jsx
const IP = "192.168.1.42"; // <-- your IPv4 from ipconfig
```

---

# ▶️ 6. Start the Expo Metro bundler

From `TheVisionariesFrontend`:

```bash
# start the dev server
expo start
# or older CLI:
# expo start

```

This opens the Expo dev tools in your browser.

---

# 📱 7. Open the app

- Install **Expo Go** on your phone (Play Store / App Store).
- Scan the QR code shown in Expo DevTools (or in the terminal).
- Or click `Run on Android device/emulator` / `Run on iOS simulator` from DevTools.

---

# Group Members:

- Ali Sharjeel
- Syeda Aliza Ayaz
- Ghulam Fatima
