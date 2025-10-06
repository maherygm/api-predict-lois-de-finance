import "dotenv/config";
import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import multer from "multer";
import path from "path";
import { ChatPromptTemplate } from "@langchain/core/prompts";

import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "@langchain/core/documents";
import { RetrievalQAChain } from "langchain/chains";
import { execFile } from "child_process";
import fs from "fs";
import XLSX from "xlsx";

// ---------- Multer setup ----------
const upload = multer({ dest: "uploads/" });

// ---------- LLM Factory ----------
class LLMFactory {
  static createGemini() {
    return new ChatGoogleGenerativeAI({
      model: "gemini-2.5-flash",
      temperature: 0.3,
      apiKey: process.env.GOOGLE_API_KEY,
    });
  }
}

// ---------- Embeddings Factory ----------
class EmbeddingsFactory {
  static createGemini() {
    return new GoogleGenerativeAIEmbeddings({
      model: "gemini-embedding-exp-03-07",
      apiKey: process.env.GOOGLE_API_KEY,
    });
  }
}

class VectorStoreFactory {
  static async create(filePath = "./data/loi_finance.xlsx") {
    // 1️⃣ Charger le fichier Excel
    const workbook = XLSX.readFile(filePath);
    const sheetName = workbook.SheetNames[0];
    const sheet = workbook.Sheets[sheetName];
    const jsonData = XLSX.utils.sheet_to_json(sheet);

    // 2️⃣ Créer des documents LangChain pour RAG
    const docs = jsonData.map(
      (row) =>
        new Document({
          pageContent: `Région: ${row.Région}\nAnnée: ${row.Année}\nBudget Santé: ${row.Budget_Santé}\nPopulation: ${row.Population}\nCroissance: ${row.Croissance}\nDépenses Santé: ${row.Dépenses_Santé}`,
          metadata: { src: "loi_finance" },
        })
    );

    // 3️⃣ Découpage en chunks pour la RAG
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const splitDocs = await splitter.splitDocuments(docs);

    // 4️⃣ Créer embeddings et MemoryVectorStore
    const embeddings = EmbeddingsFactory.createGemini();
    const vectorStore = new MemoryVectorStore(embeddings);
    await vectorStore.addDocuments(splitDocs);

    return vectorStore;
  }
}

// ---------- Initialise RAG ----------
let ragChain;
async function initRAG() {
  const llm = LLMFactory.createGemini();
  const vectorStore = await VectorStoreFactory.create();
  const prompt = ChatPromptTemplate.fromTemplate(`
    Tu es un assistant intelligent. Utilise les informations pertinentes pour répondre à la question suivante.

    Contexte :
    {context}

    Question : {query}
    Réponse :
  `);
  ragChain = RetrievalQAChain.fromLLM(llm, vectorStore.asRetriever(), {
    prompt,
  });
}

// ---------- Express server ----------
async function main() {
  await initRAG();
  const app = express();
  app.use(cors());
  app.use(bodyParser.json());

  // RAG endpoint
  app.post("/ask", async (req, res) => {
    try {
      const { question } = req.body;
      if (!question)
        return res.status(400).json({ error: "question manquante" });
      const result = await ragChain.invoke({ query: question });
      res.json({ response: result.text });
    } catch (err) {
      console.error(err);
      res.status(500).json({ error: "Erreur serveur" });
    }
  });

  // Prévision / Python endpoint
  app.post("/predict", upload.single("file"), (req, res) => {
    if (!req.file) return res.status(400).json({ error: "Fichier manquant" });

    const filePath = path.resolve(req.file.path);

    execFile(
      "python3",
      ["services/merge_forecast.py", filePath],
      { maxBuffer: 1024 * 1024 * 10 }, // 10MB buffer
      (err, stdout, stderr) => {
        // Supprimer le fichier temporaire
        fs.unlinkSync(filePath);

        if (err) {
          console.error(err, stderr);
          return res.status(500).json({ error: "Erreur prévision Python" });
        }
        try {
          const data = JSON.parse(stdout);
          res.json(data);
        } catch (e) {
          console.error(e);
          res.status(500).json({ error: "Erreur parsing JSON Python" });
        }
      }
    );
  });

  app.listen(3000, () =>
    console.log("🚀 Serveur RAG + prévision sur http://localhost:3000")
  );
}

main();
