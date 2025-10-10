export async function interpretForecast(forecastData) {
  if (!forecastData) return "Aucune donnée fournie.";

  const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY, // Assure-toi que cette variable est bien définie
  });

  const model = "gemma-3-27b-it";

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

  const contents = [
    {
      role: "user",
      parts: [
        {
          text: `Tu es un expert en finances publiques. Analyse les prévisions ci-dessus et rédige une interprétation claire et bref pas trop long pour un responsable de budget.

${contextText}`,
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

  return interpretation;
}
