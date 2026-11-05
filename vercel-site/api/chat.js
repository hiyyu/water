// Vercel serverless function: keeps the xAI key on the server, never in the browser.
const SYSTEM = `You are the assistant for a final-year project website: "Prediction of Water Quality Index & Contamination Risk Using Machine Learning" for the Klang and Skudai rivers in Malaysia, using Department of Environment (DOE) sampling data.
Facts you can rely on:
- WQI is the DOE Water Quality Index, calculated from six parameters: DO, BOD, COD, SS, pH and NH3-N.
- Status bands: 81 and above = Clean (blue), 60 to 80 = Slightly Polluted (yellow), below 60 = Polluted (red).
- Models: Persistence (baseline), CEEMDAN, LSTM, XGBoost, CEEMDAN-LSTM and CEEMDAN-LSTM-XGBoost. They forecast WQI only, not individual parameters.
- Models were scored on different numbers of test samples, so metrics are indicative, not a like-for-like ranking.
- The dashboard has four tabs: Water Quality Forecast, Latest vs Next Forecast, Model Evaluation and Live Monitoring (hourly DOE MyEQMS stations, separate from the sampling stations used for training).
Rules: answer in plain text with no markdown, in under 120 words. Do not invent specific readings, dates or metric values; point the user to the relevant dashboard tab instead. Forecasts are a research prototype and must not be used for safety decisions. If asked about unrelated topics, politely steer back to water quality and this project.`;

module.exports = async (req, res) => {
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' });
  const key = process.env.GROQ_API_KEY;
  if (!key) return res.status(500).json({ error: 'The chatbot is not configured yet.' });

  const raw = Array.isArray(req.body && req.body.messages) ? req.body.messages : [];
  const msgs = raw
    .filter(m => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string')
    .slice(-10)
    .map(m => ({ role: m.role, content: m.content.slice(0, 1000) }));
  if (!msgs.length || msgs[msgs.length - 1].role !== 'user') return res.status(400).json({ error: 'Bad request.' });

  try {
    const r = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + key },
      body: JSON.stringify({ model: process.env.GROQ_MODEL || 'llama-3.3-70b-versatile', messages: [{ role: 'system', content: SYSTEM }, ...msgs] })
    });
    const d = await r.json();
    const reply = d && d.choices && d.choices[0] && d.choices[0].message && d.choices[0].message.content;
    if (!r.ok || !reply) return res.status(502).json({ error: 'The assistant is unavailable right now.' });
    return res.status(200).json({ reply });
  } catch (e) {
    return res.status(502).json({ error: 'The assistant is unavailable right now.' });
  }
};