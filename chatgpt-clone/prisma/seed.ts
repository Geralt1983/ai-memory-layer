import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

async function main() {
  const convo = await prisma.conversation.create({
    data: {
      name: "Sample chat",
      model: "gpt-4o",
      temperature: 0.7,
      systemPrompt: "You are a helpful and concise assistant."
    }
  });

  const userMessage = await prisma.message.create({
    data: {
      role: "user",
      conversationId: convo.id,
      parts: {
        create: [{
          type: "text",
          text: "Hello, how are you?"
        }]
      }
    }
  });

  const assistantMessage = await prisma.message.create({
    data: {
      role: "assistant",
      conversationId: convo.id,
      parts: {
        create: [{
          type: "text",
          text: "Hi there! I'm doing well, thank you for asking. How can I help you today?"
        }]
      }
    }
  });

  console.log("Seeded database with:");
  console.log("- Conversation:", convo.id);
  console.log("- User message:", userMessage.id);
  console.log("- Assistant message:", assistantMessage.id);
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });