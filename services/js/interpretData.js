import { GoogleGenAI } from "@google/genai";

import Groq from "groq-sdk";

export async function interpretForecast(forecastData) {
  if (!forecastData) return "Aucune donnée fournie.";

  const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY, // Assure-toi que cette variable est bien définie
  });

  const model = "gemini-2.0-flash-lite";

  // Préparer le contexte textuel

  console.log(forecastData);
  const contextText = `
Prévisions régionales :
${forecastData.forecast_regional
  .map(
    (r) =>
      `Région: ${r.Région}, Année: ${r.Année}, Budget Santé: ${r.Budget_Santé}, Population: ${r.Population}, Croissance: ${r.Croissance}, Dépenses prévues: ${r.Dépenses_Prédites}`
  )
  .join("\n")}

Prévisions nationales :
${forecastData.forecast_national
  .map((n) => `Année: ${n.Année}, Dépenses prévues: ${n.Dépenses_Prédites}`)
  .join("\n")}
`;

  function cleanAIResponse(response) {
    // Supprime les blocs de code ```html``` ou ```
    return response
      .replace(/```html/g, "")
      .replace(/```/g, "")
      .trim();
  }

  const contents = [
    {
      role: "user",
      parts: [
        {
          text: `
              Tu es un expert en finances publiques et un développeur Front-End senior.
              Ta mission : Analyser les données budgétaires ci-dessous et générer un rapport HTML visuellement impeccable, compact et moderne.

              CONSIGNES TECHNIQUES (Strictes) :
              1. Format de sortie : Uniquement du code HTML brut. PAS de balises \`\`\`html, pas de markdown, pas de texte avant ou après.
              2. CSS : Utilise du CSS inline (style="...") pour garantir un rendu parfait lors de l'injection.
              3. Design :
                - Style "Dashboard moderne" : Police sans-serif (Inter, Segoe UI, Arial), fond blanc, ombres légères.
                - Couleurs : Bleu marine (#1e3a8a) pour les titres, Gris foncé (#374151) pour le texte, une touche de vert pour les éléments positifs.
                - Espacement : COMPACT. Évite les marges inutiles. Utilise 'margin-bottom: 10px' max pour les paragraphes.
              4. Structure :
                - Enveloppe tout le contenu dans une <div style="font-family: sans-serif; line-height: 1.5; color: #333; background: #f9fafb; padding: 20px; border-radius: 8px; border: 1px solid #e5e7eb;">.
                - Titres : <h2> (gras, couleur foncée, marge basse réduite) et <h3> (semi-gras, gris).
                - Points clés : Utilise une liste <ul> avec des puces stylisées ou des icônes simples.

              CONSIGNES DE CONTENU (Expertise Finance) :
              - Analyse les tendances : Croissance vs Inflation, disparités régionales, impact démographique.
              - Ton : Professionnel mais accessible (vulgarisation intelligente).
              - Structure du texte :
                1. Synthèse Globale (2 phrases max).
                2. Analyse Détaillée (Focus Santé & Démographie).
                3. Recommandations Stratégiques (Liste à puces concrète).
                4. Conclusion.

              Voici les données à analyser :
              ${contextText}
              `,
        },
      ],
    },
  ];

  // Appel au modèle
  const responseStream = await ai.models.generateContentStream({
    model,
    config: {},
    contents,
  });

  // Lire le flux et concaténer le texte
  let interpretation = "";
  for await (const chunk of responseStream) {
    if (chunk.text) interpretation += chunk.text;
  }

  return cleanAIResponse(interpretation);
}

// Initialisation de l'API Gemini (à adapter selon votre contexte)
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// --- Fonction de Génération de Script (NOUVELLE ÉTAPE) ---
/**
 * Convertit un texte d'analyse simple en un script de dialogue pour TTS.
 * @param {string} analysisText Le texte d'analyse brut.
 * @returns {Promise<string>} Le script formaté avec des balises <Speaker X>.
 */
export async function scriptifyForPodcast(analysisText) {
  const userQuery = `
    Transforme l'analyse financière suivante en un script de dialogue dynamique et engageant entre deux analystes: 'Speaker 1' et 'Speaker 2'. 
    Speaker 1 doit poser les questions générales et introduire les points clés.
    Speaker 2 doit fournir les données spécifiques, les tendances et les conclusions.
    Le résultat DOIT être au format suivant, sans introduction ni conclusion supplémentaires, en utilisant uniquement les balises <Speaker X>:
    <Speaker 1>Bonjour, discutons de...</Speaker 1>
    <Speaker 2>Absolument, l'analyse montre que...</Speaker 2>
    
    ANALYSE À CONVERTIR:
    ---
    ${analysisText}
    ---
  `;

  // System Instruction pour guider le style
  const systemPrompt =
    "Vous êtes un expert en scénarisation audio. Votre tâche est de convertir une analyse en un dialogue structuré, utilisant exclusivement les balises <Speaker 1> et <Speaker 2> pour attribuer la parole.";

  const model = "gemini-2.0-flash-lite";
  const maxRetries = 3;
  let responseText = null;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await ai.models.generateContent({
        model: model,
        contents: [{ parts: [{ text: userQuery }] }],
        systemInstruction: { parts: [{ text: systemPrompt }] },
      });

      responseText = response.candidates?.[0]?.content?.parts?.[0]?.text;
      if (responseText) return responseText;
    } catch (error) {
      console.warn(
        `Tentative ${attempt + 1} échouée pour LLM Scriptification.`,
        error.message
      );
      if (attempt < maxRetries - 1) {
        const delay = Math.pow(2, attempt) * 1000; // Backoff exponentiel (1s, 2s, 4s)
        await new Promise((resolve) => setTimeout(resolve, delay));
      } else {
        throw new Error(
          "Échec de la conversion du texte en script après plusieurs tentatives."
        );
      }
    }
  }
  return ""; // Ne devrait pas être atteint
}

// Initialisation de l'instance Groq en dehors de la fonction pour une meilleure performance
// L'API key sera lue automatiquement depuis process.env.GROQ_API_KEY
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

export async function interpretForecastGroq(forecastData) {
  if (!forecastData) return "Aucune donnée fournie.";

  // 1. Définir le modèle Groq à utiliser.
  // 'mixtral-8x7b-instruct-v0.1' est très puissant pour le raisonnement.
  // 'llama3-8b-8192' est également un excellent choix pour la vitesse.
  const model = "llama-3.1-8b-instant";

  // Préparer le contexte textuel
  console.log(forecastData);
  const contextText = `
Prévisions régionales :
${forecastData.forecast_regional
  .map(
    (r) =>
      `Région: ${r.Région}, Année: ${r.Année}, Budget Santé: ${r.Budget_Santé}, Population: ${r.Population}, Croissance: ${r.Croissance}, Dépenses prévues: ${r.Dépenses_Prédites}`
  )
  .join("\n")}

Prévisions nationales :
${forecastData.forecast_national
  .map((n) => `Année: ${n.Année}, Dépenses prévues: ${n.Dépenses_Prédites}`)
  .join("\n")}
`;

  function cleanAIResponse(response) {
    // Supprime les blocs de code ```html``` ou ```
    return response
      .replace(/```html/g, "")
      .replace(/```/g, "")
      .trim();
  }

  // 2. Préparer le contenu du message pour l'API Groq
  const userPrompt = `
    Tu es un expert en finances publiques et un développeur Front-End senior.
    Ta mission : Analyser les données budgétaires ci-dessous et générer un rapport HTML visuellement impeccable, compact et moderne.

    CONSIGNES TECHNIQUES (Strictes) :
    1. Format de sortie : Uniquement du code HTML brut. PAS de balises \`\`\`html, pas de markdown, pas de texte avant ou après.
    2. CSS : Utilise du CSS inline (style="...") pour garantir un rendu parfait lors de l'injection.
    3. Design :
        - Style "Dashboard moderne" : Police sans-serif (Inter, Segoe UI, Arial), fond blanc, ombres légères.
        - Couleurs : Bleu marine (#1e3a8a) pour les titres, Gris foncé (#374151) pour le texte, une touche de vert pour les éléments positifs.
        - Espacement : COMPACT. Évite les marges inutiles. Utilise 'margin-bottom: 10px' max pour les paragraphes.
    4. Structure :
        - Enveloppe tout le contenu dans une <div style="font-family: sans-serif; line-height: 1.5; color: #333; background: #f9fafb; padding: 20px; border-radius: 8px; border: 1px solid #e5e7eb;">.
        - Titres : <h2> (gras, couleur foncée, marge basse réduite) et <h3> (semi-gras, gris).
        - Points clés : Utilise une liste <ul> avec des puces stylisées ou des icônes simples.

    CONSIGNES DE CONTENU (Expertise Finance) :
    - Analyse les tendances : Croissance vs Inflation, disparités régionales, impact démographique.
    - Ton : Professionnel mais accessible (vulgarisation intelligente).
    - Structure du texte :
        1. Synthèse Globale (2 phrases max).
        2. Analyse Détaillée (Focus Santé & Démographie).
        3. Recommandations Stratégiques (Liste à puces concrète).
        4. Conclusion.

    Voici les données à analyser :
    ${contextText}
  `;

  // 3. Appel à l'API Groq
  try {
    const chatCompletion = await groq.chat.completions.create({
      messages: [
        {
          role: "user",
          content: userPrompt,
        },
      ],
      model: model, // Utilisation du modèle rapide défini ci-dessus
      // Ajoutez ici d'autres options si nécessaire, comme 'temperature' ou 'max_tokens'
    });

    // 4. Extraction et Nettoyage de la réponse
    const interpretation = chatCompletion.choices[0]?.message?.content || "";
    return cleanAIResponse(interpretation);
  } catch (error) {
    console.error("Erreur lors de l'appel à l'API Groq:", error);
    // Gérer spécifiquement les erreurs de quota 429 ici si vous voulez
    return "Erreur lors de l'analyse des prévisions (Problème API).";
  }
}

// NOTE : Vous n'avez plus besoin des fonctions 'main' et 'getGroqChatCompletion'
// de l'exemple précédent car tout est intégré dans 'interpretForecast'.
