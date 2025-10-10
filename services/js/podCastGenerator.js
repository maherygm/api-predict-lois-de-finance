import "dotenv/config";
import fs from "fs";
import path from "path";
import { GoogleGenAI } from "@google/genai";
import mime from "mime";

// ---------------- WAV Utilities ----------------
function saveBinaryFile(filePath, content) {
  fs.writeFile(filePath, content, (err) => {
    if (err) return console.error(`Error writing file ${filePath}:`, err);
    console.log(`File ${filePath} saved.`);
  });
}

function convertToWav(rawData, mimeType) {
  const options = parseMimeType(mimeType);
  const wavHeader = createWavHeader(rawData.length, options);
  const buffer = Buffer.from(rawData, "base64");
  return Buffer.concat([wavHeader, buffer]);
}

function parseMimeType(mimeType) {
  const [fileType, ...params] = mimeType.split(";").map((s) => s.trim());
  const [_, format] = fileType.split("/");
  const options = { numChannels: 1, sampleRate: 44100, bitsPerSample: 16 };

  if (format && format.startsWith("L")) {
    const bits = parseInt(format.slice(1), 10);
    if (!isNaN(bits)) options.bitsPerSample = bits;
  }

  for (const param of params) {
    const [key, value] = param.split("=").map((s) => s.trim());
    if (key === "rate") options.sampleRate = parseInt(value, 10);
  }

  return options;
}

function createWavHeader(dataLength, options) {
  const { numChannels, sampleRate, bitsPerSample } = options;
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

// ---------------- Podcast Function ----------------
export async function generatePodcast(text, outputDir = "podcasts") {
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
  const model = "gemini-2.5-flash-preview-tts";
  const config = {
    temperature: 1,
    responseModalities: ["audio"],
    speechConfig: {
      multiSpeakerVoiceConfig: {
        speakerVoiceConfigs: [
          {
            speaker: "Speaker 1",
            voiceConfig: { prebuiltVoiceConfig: { voiceName: "Enceladus" } },
          },
          {
            speaker: "Speaker 2",
            voiceConfig: { prebuiltVoiceConfig: { voiceName: "Puck" } },
          },
        ],
      },
    },
  };

  const contents = [{ role: "user", parts: [{ text }] }];

  const response = await ai.models.generateContentStream({
    model,
    config,
    contents,
  });

  if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

  let fileIndex = 0;
  const files = [];

  for await (const chunk of response) {
    const inlineData = chunk?.candidates?.[0]?.content?.parts?.[0]?.inlineData;
    if (inlineData) {
      let buffer = Buffer.from(inlineData.data || "", "base64");
      let fileExt = mime.getExtension(inlineData.mimeType || "") || "wav";
      if (!mime.getExtension(inlineData.mimeType || ""))
        buffer = convertToWav(inlineData.data || "", inlineData.mimeType || "");

      const fileName = path.join(
        outputDir,
        `podcast_${fileIndex++}.${fileExt}`
      );
      saveBinaryFile(fileName, buffer);
      files.push(fileName);
    }
  }

  return files;
}
