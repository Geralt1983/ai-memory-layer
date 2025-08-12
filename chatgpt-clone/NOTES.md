Quick start for SQLite local:

1. npm install
2. npx prisma migrate dev --name init
3. npx prisma generate
4. npm run prisma:seed
5. npm run dev

Health banner:
- If the UI says "Migrations not applied" click "Run migrate"
- Or run: npm run prisma:deploy

Later if you want Postgres again:
- revert prisma/schema.prisma to provider "postgresql" and url env("DATABASE_URL")
- set DATABASE_URL in .env
- use docker compose or your own Postgres
- run prisma migrate dev