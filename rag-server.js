import "dotenv/config";
import express from "express";
import cors from "cors";

import multer from "multer";
import path from "path";
import { ChatPromptTemplate } from "@langchain/core/prompts";

import { GoogleGenAI } from "@google/genai";
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

import {
  interpretForecast,
  scriptifyForPodcast,
} from "./services/js/interpretData.js";
import { generatePodcast } from "./services/js/podCastGenerator.js";

// ---------- Multer setup ----------
//
// SET STORAGE

var STORAGE = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads");
  },

  filename: function (req, file, cb) {
    cb(null, file.fieldname + "-" + Date.now());
  },
});
const upload = multer({ storage: STORAGE });

// ---------- LLM Factory ----------
class LLMFactory {
  static createGemini() {
    return new ChatGoogleGenerativeAI({
      model: "gemini-2.0-flash-lite",
      temperature: 0.3,
      apiKey: process.env.GOOGLE_API_KEY,
    });
  }
}

// ---------- Embeddings Factory ----------
class EmbeddingsFactory {
  static createGemini() {
    return new GoogleGenerativeAIEmbeddings({
      model: "gemini-embedding-001",
      apiKey: process.env.GOOGLE_API_KEY,
    });
  }
}

class VectorStoreFactory {
  static async create(filePath = "./data/template_loi_finance.xlsx") {
    const workbook = XLSX.readFile(filePath);
    const sheetName = workbook.SheetNames[0];
    const sheet = workbook.Sheets[sheetName];
    const jsonData = XLSX.utils.sheet_to_json(sheet);

    const docs = jsonData.map(
      (row) =>
        new Document({
          pageContent: `Région: ${row.Région}\nAnnée: ${row.Année}\nBudget Santé: ${row.Budget_Santé}\nPopulation: ${row.Population}\nCroissance: ${row.Croissance}\nDépenses Santé: ${row.Dépenses_Santé}`,
          metadata: { src: "loi_finance" },
        })
    );

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    const splitDocs = await splitter.splitDocuments(docs);

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
  //app.use(bodyParser.json());
  app.use(express.json());

  app.use("/podcast", express.static(path.join(process.cwd(), "podcast")));

  // RAG endpoint
  /**
   * Endpoint POST /ask
   * Attend { question } dans le body.
   * Retourne { response: ... }
   */
  app.post("/ask", async (req, res) => {
    try {
      const { question } = req.body;
      console.log("body", req);
      if (!question) {
        return res.status(400).json({ error: "question manquante" });
      }

      // Appel IA
      const ai = new GoogleGenAI({
        apiKey: process.env.GEMINI_API_KEY,
      });

      const model = "gemma-3-27b-it";

      const prompt = `Tu es un expert en finances publiques. Réponds de façon brève et claire à la question suivante concernant des données budgétaires (lois de finances) :\n\nQuestion : ${question}\nRéponse :`;

      const contents = [
        {
          role: "user",
          parts: [{ text: prompt }],
        },
      ];

      const responseStream = await ai.models.generateContentStream({
        model,
        config: {},
        contents,
      });

      let responseText = "";
      for await (const chunk of responseStream) {
        if (chunk.text) responseText += chunk.text;
      }

      res.json({ response: responseText.trim() });
    } catch (err) {
      console.error(err);
      res.status(500).json({ error: "Erreur serveur ou IA" });
    }
  });

  // Prévision / Python endpoint
  app.post("/predict", upload.single("file"), async (req, res) => {
    if (!req.file) return res.status(400).json({ error: "Fichier manquant" });

    const filePath = path.resolve(req.file.path);

    execFile(
      "python3",
      ["services/python/merge_forecast.py", filePath],
      { maxBuffer: 1024 * 1024 * 10 }, // 10MB buffer
      async (err, stdout, stderr) => {
        // Supprimer le fichier temporaire
        fs.unlinkSync(filePath);

        if (err) {
          console.error(err, stderr);
          return res.status(500).json({ error: "Erreur prévision Python" });
        }
        try {
          const data = JSON.parse(stdout);
          // 🔹 Appel LLM pour interprétation
          const interpretation = await interpretForecast(data);

          // 🔹 Retourne les prévisions + l’interprétation
          res.json({
            forecast: data,
            interpretation,
          });
          // res.json(data);
        } catch (e) {
          console.error(e);
          res.status(500).json({ error: "Erreur parsing JSON Python" });
        }
      }
    );
  });

  // --- Endpoint Modifié ---
  app.post("/generatePodcast", express.json(), async (req, res) => {
    try {
      const { texte } = req.body;
      if (!texte) {
        return res.status(400).json({ error: "Champ 'texte' manquant" });
      }

      // 1. ÉTAPE LLM : Transformer l'analyse brute en script de dialogue balisé
      console.log(
        "Étape 1: Transformation de l'analyse en script de dialogue..."
      );
      const dialogueScript = await scriptifyForPodcast(texte);

      if (!dialogueScript) {
        return res
          .status(500)
          .json({ error: "Le LLM n'a pas pu générer le script de dialogue." });
      }

      // 2. ÉTAPE TTS : Générer l'audio à partir du script balisé
      console.log("Étape 2: Génération du podcast audio à partir du script...");

      // Ici, on appelle votre fonction generatePodcast qui est maintenant à jour
      // (En supposant que vous ayez mis à jour generatePodcast pour retourner UN seul fichier)
      const audioPath = await generatePodcast(dialogueScript, "podcast");

      res.json({
        success: true,
        message: "Podcast généré avec succès 🎧",
        file: path.basename(audioPath), // Retourne uniquement le nom du fichier pour l'URL statique
      });
    } catch (err) {
      console.error("Erreur complète du podcast:", err.message);
      res.status(500).json({
        error:
          "Erreur serveur lors de la génération du podcast: " + err.message,
      });
    }
  });

  app.listen(3000, () =>
    console.log("🚀 Serveur RAG + prévision sur http://localhost:3000")
  );
}

main();
