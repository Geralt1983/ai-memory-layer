# ChatGPT Clone

A full-featured ChatGPT clone built with Next.js, featuring streaming responses, conversation management, and a faithful recreation of the ChatGPT UX.

## Features

- 🎯 **Two-pane layout** - Narrow sidebar for conversations, wide chat area
- 💬 **Streaming responses** - Real-time token streaming with SSE
- 🗂️ **Conversation management** - Create, rename, delete, export conversations
- 🎛️ **Model controls** - Switch models, adjust temperature, system prompts
- 🌙 **Dark mode** - ChatGPT-style dark interface by default
- ⌨️ **Keyboard shortcuts** - Cmd+Enter to send, Esc to stop
- 📱 **Responsive design** - Works on desktop and mobile
- 🎨 **Markdown support** - Code blocks with syntax highlighting
- 💾 **Persistence** - Full Prisma + PostgreSQL backend
- 🔄 **Auto-migration** - One-click database migrations in dev

## Quick Start

1. **Start the database**
   ```bash
   docker compose up -d
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment**
   ```bash
   # .env file is already created with defaults
   # Add your OpenAI API key (optional - works without it in echo mode)
   echo 'OPENAI_API_KEY=your-key-here' >> .env
   ```

4. **Run migrations and seed**
   ```bash
   npx prisma migrate dev
   npx prisma generate
   npm run prisma:seed
   ```

5. **Start the development server**
   ```bash
   npm run dev
   ```

6. **Open the app**
   Visit [http://localhost:3000](http://localhost:3000)

## Architecture

### Frontend
- **Next.js 14** with App Router
- **Tailwind CSS** for styling
- **TypeScript** for type safety
- **Streaming SSE** for real-time responses

### Backend
- **Prisma ORM** with PostgreSQL
- **Edge runtime** for streaming API
- **Node.js runtime** for database operations
- **Docker PostgreSQL** for local development

### Database Schema
```prisma
model Conversation {
  id           String    @id @default(cuid())
  name         String
  model        String
  temperature  Float     @default(0.7)
  systemPrompt String?
  messages     Message[]
  createdAt    DateTime  @default(now())
  updatedAt    DateTime  @updatedAt
}

model Message {
  id             String        @id @default(cuid())
  role           String
  createdAt      DateTime      @default(now())
  conversation   Conversation  @relation(...)
  conversationId String
  parts          Part[]
}

model Part {
  id        String   @id @default(cuid())
  type      String   // "text", "code", "image"
  text      String?
  code      String?
  src       String?
  message   Message  @relation(...)
  messageId String
}
```

## API Endpoints

### Chat Streaming
- `POST /api/chat` - Stream chat responses (SSE)

### Conversations
- `GET /api/conversations/list` - List all conversations
- `POST /api/conversations/get` - Get conversation with messages
- `POST /api/conversations/create` - Create new conversation
- `POST /api/conversations/rename` - Rename conversation
- `POST /api/conversations/delete` - Delete conversation
- `POST /api/conversations/appendMessage` - Add message
- `POST /api/conversations/updateMeta` - Update model/temperature
- `POST /api/conversations/setSystemPrompt` - Set system prompt

### System
- `GET /api/health` - Health check and migration status
- `POST /api/migrate` - Run database migrations (dev only)

## Development

### Without OpenAI API Key
The app works in "echo mode" without an API key - it will simulate streaming responses for testing the UI.

### With OpenAI API Key
Set `OPENAI_API_KEY` in `.env` to use real OpenAI responses.

### Database Management
```bash
# View data
npx prisma studio

# Reset database
npx prisma migrate reset

# Deploy to production
npx prisma migrate deploy
```

### Scripts
```bash
npm run dev          # Start with database
npm run dev:next     # Start Next.js only
npm run db:up        # Start database
npm run db:down      # Stop database
npm run prisma:seed  # Seed database
```

## Keyboard Shortcuts

- `Cmd/Ctrl + Enter` - Send message
- `Cmd/Ctrl + K` - Search conversations (coming soon)
- `Esc` - Stop streaming response
- `Shift + Enter` - New line in message

## Production Deployment

1. Set production environment variables:
   ```bash
   DATABASE_URL="your-production-db-url"
   OPENAI_API_KEY="your-openai-key"
   # DO NOT set ALLOW_DB_MIGRATION_API in production
   ```

2. Run migrations before starting:
   ```bash
   npx prisma migrate deploy
   ```

3. Build and start:
   ```bash
   npm run build
   npm start
   ```

## License

MIT