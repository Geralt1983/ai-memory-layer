# Cerebro Apocalypse - X-Men Interface

A Next.js frontend featuring the Cerebro Apocalypse design - a hybrid X-Men interface that combines the classic Cerebro system with Apocalypse font styling.

## 🦸‍♂️ Features

- **Complete Apocalypse Font Styling**: Every element uses the creepy "Creepster" font
- **Dark Apocalyptic Background**: Red/black gradients with apocalyptic animations
- **X-Men Gold Accents**: Classic X-Men gold borders and highlights
- **Interactive Chat Interface**: Full conversation functionality with typing indicators
- **Responsive Design**: Works on desktop and mobile devices
- **Mission Control**: Character selector and mission management
- **Apocalyptic Effects**: Theme toggle with dramatic visual effects

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Navigate to the frontend directory:
```bash
cd chatgpt-clone-frontend
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Run the development server:
```bash
npm run dev
# or
yarn dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## 🎨 Design Elements

### Typography
- **Headers**: Apocalypse "Creepster" font for dramatic effect
- **Body Text**: Apocalypse font throughout the interface
- **Buttons**: Apocalypse font with uppercase styling

### Colors
- **Primary**: Dark apocalyptic background (#0a0a0a)
- **Secondary**: Red-tinted panels (rgba(20, 0, 0, 0.9))
- **Accent**: X-Men gold (#ffd700) and orange (#ff6b35)
- **Text**: White primary, blue-tinted muted text

### Animations
- **Apocalyptic Shift**: Background animation with hue rotation
- **Power Flow**: Animated borders and power meters
- **Comic Slide**: Message entry animations
- **Mutant Glow**: Interactive element highlights

## 🛠️ Tech Stack

- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling with custom theme
- **Lucide React**: Icon library
- **Google Fonts**: Creepster, Special Elite, Orbitron

## 📱 Responsive Features

- **Desktop**: Full sidebar and main content layout
- **Mobile**: Collapsible sidebar with overlay
- **Touch-friendly**: Optimized for mobile interactions

## 🎯 Usage

1. **Start a New Mission**: Click "NEW MISSION" to begin fresh
2. **Select Character**: Choose from Professor X, Cyclops, Wolverine, or Storm
3. **Send Messages**: Type and press Enter to communicate with Cerebro
4. **Toggle Powers**: Click "TOGGLE POWERS" for apocalyptic effects
5. **View History**: Access previous missions in the sidebar

## 🔧 Customization

The design can be easily customized by modifying:
- `app/globals.css`: Main styling and animations
- `tailwind.config.js`: Color palette and theme extensions
- `components/CerebroChat.tsx`: Main interface logic

## 🚀 Deployment

Build for production:
```bash
npm run build
npm start
```

Deploy to Vercel:
```bash
vercel
```

## 🦸‍♂️ X-Men Integration

This interface is designed to work with your existing AI memory layer backend. You can integrate it by:

1. Connecting to your API endpoints
2. Replacing the mock AI responses with real backend calls
3. Adding authentication for different X-Men characters
4. Implementing real-time mission updates

---

**Welcome to the X-Men, Mutant!** 🚀
