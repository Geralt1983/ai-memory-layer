# Multi-stage build for ChatGPT clone with memory layer
FROM node:20-alpine AS base
WORKDIR /app

# Install dependencies
FROM base AS deps
COPY package*.json ./
RUN npm ci --only=production

# Build the Next.js app
FROM base AS builder
COPY package*.json ./
COPY prisma ./prisma
RUN npm ci
COPY . .
RUN npm run build

# Python memory API stage
FROM python:3.11-slim AS python-deps
WORKDIR /app
COPY requirements.txt ./
COPY ../requirements.txt ../requirements.txt 2>/dev/null || echo "No parent requirements.txt"
RUN pip install --no-cache-dir -r requirements.txt
# Install parent project dependencies if available
RUN pip install --no-cache-dir fastapi uvicorn pydantic python-dotenv

# Production image
FROM base AS runner
WORKDIR /app

# Install Python for memory API
RUN apk add --no-cache python3 py3-pip sqlite

ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy Next.js app
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/prisma ./prisma

# Copy Python memory components
COPY --chown=nextjs:nodejs lib/ ./lib/
COPY --chown=nextjs:nodejs start_memory_api.py ./
COPY --chown=nextjs:nodejs requirements.txt ./

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Create data directory and set permissions
RUN mkdir -p /app/data && chown nextjs:nodejs /app/data
RUN mkdir -p /app/prisma && chown nextjs:nodejs /app/prisma

USER nextjs

EXPOSE 3000 8001

# Start both Next.js and Python memory API
CMD ["sh", "-c", "python3 start_memory_api.py & node server.js"]