// services/podcastGenerator.js
import { GoogleGenAI } from "@google/genai";
import mime from "mime";
import fs from "fs";
import path from "path";

/**
 * Convert raw audio data to WAV if needed
 */
function createWavHeader(
  dataLength,
  { numChannels, sampleRate, bitsPerSample }
) {
  const byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const buffer = Buffer.alloc(44);

  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + dataLength, 4);
  buffer.write("WAVE", 8);
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(numChannels, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(byteRate, 28);
  buffer.writeUInt16LE(blockAlign, 32);
  buffer.writeUInt16LE(bitsPerSample, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(dataLength, 40);

  return buffer;
}

function parseMimeType(mimeType) {
  const [fileType, ...params] = mimeType.split(";").map((s) => s.trim());
  const [_, format] = fileType.split("/");

  const options = { numChannels: 1, sampleRate: 44100, bitsPerSample: 16 };

  for (const param of params) {
    const [key, value] = param.split("=").map((s) => s.trim());
    if (key === "rate") options.sampleRate = parseInt(value, 10);
  }

  return options;
}

/**
 * Sauvegarde un fichier audio sur le disque
 */
function saveBinaryFile(fileName, content) {
  const outputDir = path.resolve("podcasts");
  if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir);

  const filePath = path.join(outputDir, fileName);
  fs.writeFileSync(filePath, content);
  console.log(`🎧 Fichier audio sauvegardé : ${filePath}`);
  return filePath;
}

/**
 * Fonction principale de génération audio
 */
export async function generatePodcastAudio(text, filePrefix = "podcast") {
  try {
    const ai = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY,
    });

    const config = {
      temperature: 1,
      responseModalities: ["audio"],
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: {
            voiceName: "Zephyr", // autre voix : "Wave", "Serene", "Astra"...
          },
        },
      },
    };

    const model = "gemini-2.5-pro-preview-tts";
    const contents = [
      {
        role: "user",
        parts: [
          {
            text,
          },
        ],
      },
    ];

    const response = await ai.models.generateContentStream({
      model,
      config,
      contents,
    });

    let fileIndex = 0;
    let lastFilePath = null;

    for await (const chunk of response) {
      const part = chunk?.candidates?.[0]?.content?.parts?.[0];
      if (part?.inlineData) {
        const inlineData = part.inlineData;
        let fileExtension =
          mime.getExtension(inlineData.mimeType || "") || "wav";
        const buffer = Buffer.from(inlineData.data || "", "base64");

        const fileName = `${filePrefix}-${Date.now()}-${fileIndex++}.${fileExtension}`;
        lastFilePath = saveBinaryFile(fileName, buffer);
      }
    }

    return lastFilePath;
  } catch (error) {
    console.error("❌ Erreur génération audio :", error);
    throw error;
  }
}
